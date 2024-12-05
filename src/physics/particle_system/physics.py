import numpy as np
from numba import cuda, float32, int32, jit
import math
from core.constants import (
    GRAVITY, WINDOW_HEIGHT, WINDOW_WIDTH, COLLISION_RESPONSE, 
    FIXED_TIMESTEP, MAX_VELOCITY, PARTICLE_RADIUS, FLOOR_BUFFER, MIN_BOUNCE_VELOCITY,
    SIMULATION_HEIGHT, SIMULATION_WIDTH, SIMULATION_Y_OFFSET, SIMULATION_X_OFFSET,
    ELECTROMAGNETIC_CONSTANT, DRAG_FORCE_MULTIPLIER
)
from utils.profiler import profile_function
from physics.particle_system.collision import CollisionHandler
from physics.particle_system.gpu_accelerator import update_positions_gpu, update_velocities_gpu, handle_collisions_gpu


# Physics constants
CUDA_MAX_VELOCITY = 20.0
CUDA_GRAVITY = 4.0
CUDA_COLLISION_RESPONSE = 1.0
THREADS_PER_BLOCK = 128
MAX_BLOCKS = 256
MIN_SEPARATION = PARTICLE_RADIUS * 2.2  # Slightly larger than diameter to ensure no overlap

@cuda.jit
def update_positions_gpu(positions, velocities, dt):
    idx = cuda.grid(1)
    if idx < positions.shape[0]:
        positions[idx, 0] = positions[idx, 0] + velocities[idx, 0] * dt
        positions[idx, 1] = positions[idx, 1] + velocities[idx, 1] * dt

@cuda.jit
def update_velocities_gpu(velocities, dt):
    idx = cuda.grid(1)
    if idx < velocities.shape[0]:
        velocities[idx, 1] += CUDA_GRAVITY * dt
        velocities[idx, 1] = min(max(velocities[idx, 1], -CUDA_MAX_VELOCITY), CUDA_MAX_VELOCITY)
        velocities[idx, 0] = min(max(velocities[idx, 0], -CUDA_MAX_VELOCITY), CUDA_MAX_VELOCITY)

@cuda.jit
def handle_collisions_gpu(positions, velocities, radii, active_particles):
    """Handle collisions between particles on GPU."""
    particle_idx = cuda.grid(1)
    if particle_idx >= active_particles:
        return

    pos_x = positions[particle_idx, 0]
    pos_y = positions[particle_idx, 1]
    vel_x = velocities[particle_idx, 0]
    vel_y = velocities[particle_idx, 1]
    my_radius = radii[particle_idx]
    
    # Check collisions with other particles
    for other_idx in range(active_particles):
        if particle_idx != other_idx:
            dx = pos_x - positions[other_idx, 0]
            dy = pos_y - positions[other_idx, 1]
            dist_sq = dx * dx + dy * dy
            
            min_dist = my_radius + radii[other_idx]
            if dist_sq < min_dist * min_dist and dist_sq > 0:
                dist = math.sqrt(dist_sq)
                nx = dx / dist
                ny = dy / dist
                
                # Immediate position correction
                overlap = min_dist - dist
                pos_x += nx * overlap * 0.5
                pos_y += ny * overlap * 0.5
                
                # Elastic collision: swap velocities along the normal
                other_vel_x = velocities[other_idx, 0]
                other_vel_y = velocities[other_idx, 1]
                
                vel_x, other_vel_x = other_vel_x, vel_x
                vel_y, other_vel_y = other_vel_y, vel_y
                
                # Update other particle's velocity
                velocities[other_idx, 0] = other_vel_x
                velocities[other_idx, 1] = other_vel_y
    
    # Write back updated position and velocity
    positions[particle_idx, 0] = pos_x
    positions[particle_idx, 1] = pos_y
    velocities[particle_idx, 0] = vel_x
    velocities[particle_idx, 1] = vel_y

@jit(nopython=True)
def calculate_particle_density(positions, active_particles, grid_size=32):
    """Calculate particle density for adaptive timesteps"""
    density_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    cell_size_x = SIMULATION_WIDTH / grid_size
    cell_size_y = SIMULATION_HEIGHT / grid_size
    
    for i in range(active_particles):
        x = int(positions[i, 0] / cell_size_x)
        y = int(positions[i, 1] / cell_size_y)
        if 0 <= x < grid_size and 0 <= y < grid_size:
            density_grid[y, x] += 1
            
    return np.max(density_grid)

class PhysicsHandler:
    def __init__(self):
        self.use_gpu = False
        self.density_threshold = 10
        self.min_substeps = 2
        self.max_substeps = 6
        self.gpu_batch_threshold = 32
        self.dragged_particle = None
        self.drag_offset = None
        self.drag_force_multiplier = 5.0
        self.particle_radii = None

        self.grid_cell_size = 3.0 * 2.5  # slightly larger than max interaction distance
        self.grid_cells_x = int(np.ceil(SIMULATION_WIDTH / self.grid_cell_size))
        self.grid_cells_y = int(np.ceil(SIMULATION_HEIGHT / self.grid_cell_size))

        self._init_gpu()

    def _init_gpu(self):
        try:
            cuda.select_device(0)
            self.use_gpu = True
            self.threads_per_block = 128
            self.max_blocks = 256
            
            # Warm up kernels
            dummy_size = 128
            dummy_positions = np.zeros((dummy_size, 2), dtype=np.float32)
            dummy_velocities = np.zeros((dummy_size, 2), dtype=np.float32)
            
            blocks = (dummy_size + self.threads_per_block - 1) // self.threads_per_block
            pos_dev = cuda.to_device(dummy_positions)
            vel_dev = cuda.to_device(dummy_velocities)
            update_positions_gpu[blocks, self.threads_per_block](pos_dev, vel_dev, 0.016)
            update_velocities_gpu[blocks, self.threads_per_block](vel_dev, 0.016)
        except Exception as e:
            print(f"CUDA device not available, using CPU: {e}")
            self.use_gpu = False

    def _build_spatial_hash(self, positions):
        """Build a spatial hash for particles to reduce collision checks.

        Returns:
        - particle_indices: array of particle indices sorted by cell
        - cell_start: start index of each cell in particle_indices
        - cell_end: end index of each cell in particle_indices
        """
        num_particles = positions.shape[0]
        # Compute cell indices for each particle
        cell_indices = np.empty(num_particles, dtype=np.int32)
        for i in range(num_particles):
            x = int(positions[i, 0] / self.grid_cell_size)
            y = int(positions[i, 1] / self.grid_cell_size)
            # Clamp indices
            if x < 0: x = 0
            if x >= self.grid_cells_x: x = self.grid_cells_x - 1
            if y < 0: y = 0
            if y >= self.grid_cells_y: y = self.grid_cells_y - 1
            cell_idx = y * self.grid_cells_x + x
            cell_indices[i] = cell_idx

        # Sort particles by cell index
        sorted_indices = np.argsort(cell_indices)
        particle_indices = sorted_indices.astype(np.int32)

        # Build prefix arrays
        cell_start = np.full(self.grid_cells_x * self.grid_cells_y, -1, dtype=np.int32)
        cell_end = np.full(self.grid_cells_x * self.grid_cells_y, -1, dtype=np.int32)

        # Assign start/end indices
        if num_particles > 0:
            current_cell = cell_indices[particle_indices[0]]
            cell_start[current_cell] = 0

            for i in range(1, num_particles):
                cell_id = cell_indices[particle_indices[i]]
                prev_cell_id = cell_indices[particle_indices[i - 1]]
                if cell_id != prev_cell_id:
                    cell_end[prev_cell_id] = i
                    cell_start[cell_id] = i

            last_cell_id = cell_indices[particle_indices[-1]]
            cell_end[last_cell_id] = num_particles

        return particle_indices, cell_start, cell_end

    def _handle_boundaries_vectorized(self, system):
        active_slice = slice(system.active_particles)
        positions = system.positions[active_slice]
        velocities = system.velocities[active_slice]

        left_boundary = PARTICLE_RADIUS * 2.0
        right_boundary = SIMULATION_WIDTH - PARTICLE_RADIUS * 4.0
        floor_level = SIMULATION_HEIGHT - PARTICLE_RADIUS

        x_left_violation = positions[:, 0] < left_boundary
        x_right_violation = positions[:, 0] > right_boundary

        positions[x_left_violation, 0] = left_boundary
        positions[x_right_violation, 0] = right_boundary
        velocities[x_left_violation, 0] *= -COLLISION_RESPONSE * 1.2
        velocities[x_right_violation, 0] *= -COLLISION_RESPONSE * 1.2

        y_floor_violation = positions[:, 1] > floor_level
        positions[y_floor_violation, 1] = floor_level

        floor_collision = y_floor_violation & (velocities[:, 1] > 0)
        if np.any(floor_collision):
            velocities[floor_collision, 1] *= -COLLISION_RESPONSE
            velocities[floor_collision, 0] *= 0.95

        floor_contact = positions[:, 1] >= floor_level - 0.1
        slow_velocity = np.abs(velocities[:, 1]) < MIN_BOUNCE_VELOCITY
        stop_mask = floor_contact & slow_velocity
        velocities[stop_mask, 1] = 0
        positions[stop_mask, 1] = floor_level

    def _update_gpu(self, system, dt):
        active_slice = slice(system.active_particles)
        num_particles = system.active_particles
        if num_particles == 0:
            return

        threads_per_block = self.threads_per_block
        blocks = (num_particles + threads_per_block - 1) // threads_per_block

        if blocks == 0:
            self._update_cpu(system, dt)
            return

        # Update radii array if needed
        if self.particle_radii is None or len(self.particle_radii) < system.active_particles:
            radii = []
            for i in range(num_particles):
                chem = system.chemical_properties[i]
                radius = chem.element_data.radius * 20.0
                radii.append(radius)
            self.particle_radii = np.array(radii, dtype=np.float32)
        else:
            # Update existing radii if necessary
            for i in range(num_particles):
                chem = system.chemical_properties[i]
                self.particle_radii[i] = chem.element_data.radius * 20.0

        positions_f32 = system.positions[active_slice].astype(np.float32, copy=False)
        velocities_f32 = system.velocities[active_slice].astype(np.float32, copy=False)

        # Build spatial hash on CPU
        particle_indices, cell_start, cell_end = self._build_spatial_hash(positions_f32)

        # Copy data to device
        positions_gpu = cuda.to_device(positions_f32)
        velocities_gpu = cuda.to_device(velocities_f32)
        radii_gpu = cuda.to_device(self.particle_radii)
        particle_indices_gpu = cuda.to_device(particle_indices)
        cell_start_gpu = cuda.to_device(cell_start)
        cell_end_gpu = cuda.to_device(cell_end)

        update_velocities_gpu[blocks, threads_per_block](velocities_gpu, dt)
        update_positions_gpu[blocks, threads_per_block](positions_gpu, velocities_gpu, dt)

        handle_collisions_gpu[blocks, threads_per_block](
            positions_gpu, velocities_gpu, radii_gpu, num_particles,
            particle_indices_gpu, cell_start_gpu, cell_end_gpu,
            self.grid_cells_x, self.grid_cells_y,
            float32(self.grid_cell_size), float32(self.grid_cell_size)
        )

        # Copy results back
        positions_gpu.copy_to_host(system.positions[active_slice])
        velocities_gpu.copy_to_host(system.velocities[active_slice])

        self._handle_boundaries_vectorized(system)

    def _handle_boundaries_vectorized(self, system):
        active_slice = slice(system.active_particles)
        positions = system.positions[active_slice]
        velocities = system.velocities[active_slice]
        
        left_boundary = PARTICLE_RADIUS * 2.0
        right_boundary = SIMULATION_WIDTH - PARTICLE_RADIUS * 4.0
        floor_level = SIMULATION_HEIGHT - PARTICLE_RADIUS
        
        x_left_violation = positions[:, 0] < left_boundary
        x_right_violation = positions[:, 0] > right_boundary
        
        positions[x_left_violation, 0] = left_boundary
        positions[x_right_violation, 0] = right_boundary
        velocities[x_left_violation, 0] *= -COLLISION_RESPONSE * 1.2
        velocities[x_right_violation, 0] *= -COLLISION_RESPONSE * 1.2
        
        y_floor_violation = positions[:, 1] > floor_level
        positions[y_floor_violation, 1] = floor_level
        
        floor_collision = y_floor_violation & (velocities[:, 1] > 0)
        if np.any(floor_collision):
            velocities[floor_collision, 1] *= -COLLISION_RESPONSE
            velocities[floor_collision, 0] *= 0.95
        
        # Stop very slow particles on floor
        floor_contact = positions[:, 1] >= floor_level - 0.1
        slow_velocity = np.abs(velocities[:, 1]) < MIN_BOUNCE_VELOCITY
        stop_mask = floor_contact & slow_velocity
        velocities[stop_mask, 1] = 0
        positions[stop_mask, 1] = floor_level
    
    def start_drag(self, system, pos):
        pos = np.array(pos, dtype=np.float32)
        
        closest_dist = float('inf')
        closest_idx = None
        
        for i in range(system.active_particles):
            if not system.active_mask[i]:
                continue
            
            dist = np.linalg.norm(system.positions[i] - pos)
            if dist < closest_dist and dist < 20.0:
                closest_dist = dist
                closest_idx = i
        
        if closest_idx is not None:
            self.dragged_particle = closest_idx
            self.drag_offset = system.positions[closest_idx] - pos
            self.dragged_group = self._get_bonded_group(system, closest_idx)

    def _get_bonded_group(self, system, start_idx):
        visited = set()
        to_visit = {start_idx}
        
        while to_visit:
            current = to_visit.pop()
            visited.add(current)
            
            if current in system.chemical_properties:
                chem = system.chemical_properties[current]
                if hasattr(chem, 'bonds'):
                    for bond in chem.bonds:
                        if hasattr(bond, 'particle_id'):
                            bonded_idx = bond.particle_id
                            if bonded_idx not in visited:
                                to_visit.add(bonded_idx)
        
        return visited

    def update_drag(self, system, pos):
        if self.dragged_particle is None:
            return

        pos = np.array(pos, dtype=np.float32)
        
        if hasattr(self, 'dragged_group'):
            new_primary_pos = pos + self.drag_offset
            delta = new_primary_pos - system.positions[self.dragged_particle]
            
            for idx in self.dragged_group:
                system.positions[idx] += delta
                system.velocities[idx] = np.zeros(2, dtype=np.float32)

    def end_drag(self, system):
        if self.dragged_particle is not None:
            system.velocities[self.dragged_particle] = np.zeros(2, dtype=np.float32)
            self.dragged_particle = None
            self.drag_offset = None

    @profile_function(threshold_ms=1.0)
    def update(self, system, delta_time):
        if system.active_particles == 0:
            return

        if self.dragged_particle is not None:
            mask = np.ones(system.active_particles, dtype=bool)
            mask[self.dragged_particle] = False
            active_slice = mask
        else:
            active_slice = slice(system.active_particles)

        if system.active_particles < self.gpu_batch_threshold:
            self._update_cpu_simple(system, delta_time)
            return
            
        if self.use_gpu:
            particle_density = calculate_particle_density(system.positions, system.active_particles)
            density_based_substeps = max(1, int(particle_density / self.density_threshold))
            substeps = max(self.min_substeps, min(density_based_substeps, self.max_substeps))
        else:
            substeps = self.min_substeps
            
        dt = delta_time / substeps
        
        for _ in range(substeps):
            if self.use_gpu and system.active_particles >= self.gpu_batch_threshold:
                self._update_gpu(system, dt)
            else:
                self._update_cpu(system, dt)

    def _update_gpu(self, system, dt):
        active_slice = slice(system.active_particles)
        
        threads_per_block = self.threads_per_block
        num_particles = system.active_particles
        blocks = (num_particles + threads_per_block - 1) // threads_per_block
        
        if blocks == 0:
            self._update_cpu(system, dt)
            return
        
        # Update radii array if needed
        if self.particle_radii is None or len(self.particle_radii) < system.active_particles:
            radii = []
            for i in range(system.active_particles):
                chem = system.chemical_properties[i]
                radius = chem.element_data.radius * 20.0
                radii.append(radius)
            self.particle_radii = np.array(radii, dtype=np.float32)
            self.radii_gpu = cuda.to_device(self.particle_radii)
        else:
            # Update existing radii if necessary
            for i in range(system.active_particles):
                chem = system.chemical_properties[i]
                self.particle_radii[i] = chem.element_data.radius * 20.0
            self.radii_gpu.copy_to_device(self.particle_radii)
        
        # Ensure correct dtype
        positions_f32 = system.positions[active_slice].astype(np.float32, copy=False)
        velocities_f32 = system.velocities[active_slice].astype(np.float32, copy=False)
        
        if not hasattr(self, 'positions_gpu') or self.positions_gpu.size != positions_f32.size:
            self.positions_gpu = cuda.to_device(positions_f32)
            self.velocities_gpu = cuda.to_device(velocities_f32)
        else:
            self.positions_gpu.copy_to_device(positions_f32)
            self.velocities_gpu.copy_to_device(velocities_f32)
        
        update_velocities_gpu[blocks, threads_per_block](self.velocities_gpu, dt)
        update_positions_gpu[blocks, threads_per_block](self.positions_gpu, self.velocities_gpu, dt)
        handle_collisions_gpu[blocks, threads_per_block](
            self.positions_gpu, 
            self.velocities_gpu,
            self.radii_gpu,
            system.active_particles
        )
        
        self.positions_gpu.copy_to_host(system.positions[active_slice])
        self.velocities_gpu.copy_to_host(system.velocities[active_slice])
        
        self._handle_boundaries_vectorized(system)

    def _update_cpu_simple(self, system, dt):
        active_slice = slice(system.active_particles)
        
        system.velocities[active_slice, 1] += GRAVITY * dt
        system.velocities[active_slice] = np.clip(system.velocities[active_slice], -MAX_VELOCITY, MAX_VELOCITY)
        
        system.positions[active_slice] += system.velocities[active_slice] * dt
        self._handle_boundaries_simple(system)

    def _handle_boundaries_simple(self, system):
        active_slice = slice(system.active_particles)
        positions = system.positions[active_slice]
        velocities = system.velocities[active_slice]
        
        floor_level = SIMULATION_HEIGHT - PARTICLE_RADIUS
        
        floor_collision = positions[:, 1] > floor_level
        positions[floor_collision, 1] = floor_level
        velocities[floor_collision, 1] *= -COLLISION_RESPONSE
        
        left_wall = positions[:, 0] < 0
        right_wall = positions[:, 0] > SIMULATION_WIDTH - PARTICLE_RADIUS
        
        positions[left_wall, 0] = 0
        positions[right_wall, 0] = SIMULATION_WIDTH - PARTICLE_RADIUS
        
        velocities[left_wall, 0] *= -COLLISION_RESPONSE
        velocities[right_wall, 0] *= -COLLISION_RESPONSE

    def _update_cpu(self, system, delta_time):
        active_slice = slice(system.active_particles)
        
        system.active_mask[active_slice] &= (
            (system.positions[active_slice, 1] < SIMULATION_HEIGHT) &
            (system.positions[active_slice, 0] >= 0) &
            (system.positions[active_slice, 0] < SIMULATION_WIDTH)
        )
        
        if not hasattr(self, '_predicted_positions') or self._predicted_positions.shape != system.positions[active_slice].shape:
            self._predicted_positions = np.empty_like(system.positions[active_slice])
        
        system.velocities[active_slice, 1] = np.clip(
            system.velocities[active_slice, 1] + GRAVITY * delta_time,
            -MAX_VELOCITY, MAX_VELOCITY
        )
        
        np.add(system.positions[active_slice], 
               system.velocities[active_slice] * delta_time, 
               out=self._predicted_positions)
        
        floor_level = SIMULATION_HEIGHT - PARTICLE_RADIUS
        floor_collision = self._predicted_positions[:, 1] >= floor_level
        
        if np.any(floor_collision):
            self._predicted_positions[floor_collision, 1] = floor_level
            system.velocities[active_slice][floor_collision, 1] *= -COLLISION_RESPONSE
            system.velocities[active_slice][floor_collision, 0] *= 0.98
        
        system.positions[active_slice] = self._predicted_positions
        self._handle_boundaries_vectorized(system)
        self._handle_collisions_vectorized(system)

    def _handle_collisions_vectorized(self, system):
        positions = system.positions[:system.active_particles]
        velocities = system.velocities[:system.active_particles]
        
        # Build radii array
        radii = np.empty(system.active_particles, dtype=np.float32)
        for i in range(system.active_particles):
            chem = system.chemical_properties[i]
            base_radius = chem.element_data.radius * 20.0
            if chem.element_data.id == 'H':
                base_radius *= 0.6
            radii[i] = base_radius
        
        # O(n²) collision checks - consider spatial partitioning if performance is an issue
        for i in range(system.active_particles):
            for j in range(i + 1, system.active_particles):
                if self._are_particles_bonded(system, i, j):
                    self._enforce_bond_distance(system, i, j)
                    continue
                
                diff = positions[i] - positions[j]
                dist_sq = np.sum(diff * diff)
                min_dist = (radii[i] + radii[j]) * 1.1
                if dist_sq < min_dist * min_dist and dist_sq > 0:
                    dist = np.sqrt(dist_sq)
                    
                    rel_vel = velocities[i] - velocities[j]
                    collision_energy = 0.5 * np.sum(rel_vel * rel_vel)
                    
                    # Try chemical bonding
                    if CollisionHandler.handle_particle_collision(system, i, j):
                        # If bonded, unify velocities
                        avg_v = (velocities[i] + velocities[j]) * 0.5
                        velocities[i] = avg_v.copy()
                        velocities[j] = avg_v.copy()
                        continue
                    
                    # Otherwise, physical collision
                    normal = diff / dist
                    overlap = min_dist - dist
                    positions[i] += normal * overlap * 0.6
                    positions[j] -= normal * overlap * 0.6
                    
                    vel_along_normal = np.dot(rel_vel, normal)
                    if vel_along_normal > 0:
                        restitution = 0.8
                        impulse = normal * vel_along_normal * restitution
                        
                        mass_i = system.chemical_properties[i].element_data.mass
                        mass_j = system.chemical_properties[j].element_data.mass
                        total_mass = mass_i + mass_j
                        
                        velocities[i] -= impulse * (mass_j / total_mass)
                        velocities[j] += impulse * (mass_i / total_mass)

    def _are_particles_bonded(self, system, idx1, idx2):
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        if not hasattr(chem1, 'bonds') or not hasattr(chem2, 'bonds'):
            return False
        
        for bond in chem1.bonds:
            if hasattr(bond, 'particle_id') and bond.particle_id == idx2:
                return True
        return False

    def _enforce_bond_distance(self, system, idx1, idx2):
        pos1 = system.positions[idx1]
        pos2 = system.positions[idx2]
        
        radius1 = system.chemical_properties[idx1].element_data.radius
        radius2 = system.chemical_properties[idx2].element_data.radius
        ideal_distance = (radius1 + radius2) * 20.0
        
        diff = pos1 - pos2
        current_distance = np.linalg.norm(diff)
        
        if current_distance == 0:
            return
        
        direction = diff / current_distance
        correction = (ideal_distance - current_distance) * 0.5
        system.positions[idx1] += direction * correction
        system.positions[idx2] -= direction * correction
        
        avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
        system.velocities[idx1] = avg_velocity.copy()
        system.velocities[idx2] = avg_velocity.copy()

    @profile_function(threshold_ms=1.0)
    def update_collisions(self, system, delta_time):
        # Removed excessive print statements for performance
        # If needed, you can add logging here at a reduced frequency.
        for i in range(system.active_particles):
            for j in range(i + 1, system.active_particles):
                pos_diff = system.positions[i] - system.positions[j]
                distance = np.linalg.norm(pos_diff)
                
                radius_i = system.chemical_properties[i].element_data.radius
                radius_j = system.chemical_properties[j].element_data.radius
                combined_radius = (radius_i + radius_j) * 20.0
                
                # Check for collision with a lenient threshold
                if distance < combined_radius * 2.0:
                    CollisionHandler.handle_particle_collision(system, i, j)
