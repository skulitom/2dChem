import numpy as np
from numba import cuda, float32, int32, jit
import math
from core.constants import (
    GRAVITY, WINDOW_HEIGHT, WINDOW_WIDTH, COLLISION_RESPONSE, 
    FIXED_TIMESTEP, MAX_VELOCITY, PARTICLE_RADIUS, FLOOR_BUFFER, MIN_BOUNCE_VELOCITY,
    SIMULATION_HEIGHT, SIMULATION_WIDTH, SIMULATION_Y_OFFSET, SIMULATION_X_OFFSET
)
from utils.profiler import profile_function

# Optimized CUDA constants
CUDA_MAX_VELOCITY = 20.0
CUDA_GRAVITY = 4.0
CUDA_COLLISION_RESPONSE = 0.3
THREADS_PER_BLOCK = 128
MAX_BLOCKS = 256
COLLISION_DISTANCE = PARTICLE_RADIUS * 4.0  # Increased from 2.0

@cuda.jit
def update_positions_gpu(positions, velocities, dt):
    idx = cuda.grid(1)
    if idx < positions.shape[0]:
        positions[idx, 0] = cuda.fma(velocities[idx, 0], dt, positions[idx, 0])
        positions[idx, 1] = cuda.fma(velocities[idx, 1], dt, positions[idx, 1])

@cuda.jit
def update_velocities_gpu(velocities, dt):
    idx = cuda.grid(1)
    if idx < velocities.shape[0]:
        velocities[idx, 1] = cuda.fma(CUDA_GRAVITY, dt, velocities[idx, 1])
        velocities[idx, 1] = min(max(velocities[idx, 1], -CUDA_MAX_VELOCITY), CUDA_MAX_VELOCITY)
        velocities[idx, 0] = min(max(velocities[idx, 0], -CUDA_MAX_VELOCITY), CUDA_MAX_VELOCITY)

@cuda.jit
def handle_collisions_gpu(positions, velocities, active_particles):
    particle_idx = cuda.grid(1)
    if particle_idx >= active_particles:
        return

    pos_x = positions[particle_idx, 0]
    pos_y = positions[particle_idx, 1]
    vel_x = velocities[particle_idx, 0]
    vel_y = velocities[particle_idx, 1]
    
    # Check collisions with nearby particles
    for other_idx in range(active_particles):
        if particle_idx != other_idx:
            dx = pos_x - positions[other_idx, 0]
            dy = pos_y - positions[other_idx, 1]
            dist_sq = dx * dx + dy * dy
            collision_dist_sq = (PARTICLE_RADIUS * 4.0) * (PARTICLE_RADIUS * 4.0)
            
            if dist_sq < collision_dist_sq and dist_sq > 0:
                dist = math.sqrt(dist_sq)
                nx = dx / dist
                ny = dy / dist
                
                # Add repulsion force
                repulsion_strength = 0.8
                repulsion_factor = (1.0 - dist/(PARTICLE_RADIUS * 4.0)) * repulsion_strength
                vel_x += nx * repulsion_factor
                vel_y += ny * repulsion_factor
                
                dvx = vel_x - velocities[other_idx, 0]
                dvy = vel_y - velocities[other_idx, 1]
                
                vn = dvx * nx + dvy * ny
                
                if vn > 0:
                    # Stronger collision response
                    impulse = vn * CUDA_COLLISION_RESPONSE * 1.5
                    vel_x -= impulse * nx
                    vel_y -= impulse * ny
                    
                    # More aggressive separation
                    overlap = (PARTICLE_RADIUS * 4.0) - dist
                    if overlap > 0:
                        separation_factor = 0.9  # Increased from 0.7
                        pos_x += nx * overlap * separation_factor
                        pos_y += ny * overlap * separation_factor
    
    # Update final position and velocity
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
        
        # Pre-allocate arrays for collision detection
        self.cell_size = PARTICLE_RADIUS * 4
        self.grid_x = int(SIMULATION_WIDTH / self.cell_size) + 1
        self.grid_y = int(SIMULATION_HEIGHT / self.cell_size) + 1
        
        # Initialize CUDA device only when needed
        self._init_gpu()
    
    def _init_gpu(self):
        """Lazy initialization of GPU to avoid startup delay"""
        try:
            cuda.select_device(0)
            self.use_gpu = True
            self.threads_per_block = THREADS_PER_BLOCK
            self.max_blocks = MAX_BLOCKS
            
            # Pre-compile CUDA kernels with proper grid size
            dummy_size = 128  # Larger size to avoid under-utilization warning
            dummy_positions = np.zeros((dummy_size, 2), dtype=np.float32)
            dummy_velocities = np.zeros((dummy_size, 2), dtype=np.float32)
            
            blocks = (dummy_size + self.threads_per_block - 1) // self.threads_per_block
            
            # Warm up all kernels
            update_positions_gpu[blocks, self.threads_per_block](
                cuda.to_device(dummy_positions),
                cuda.to_device(dummy_velocities),
                0.016
            )
            update_velocities_gpu[blocks, self.threads_per_block](
                cuda.to_device(dummy_velocities),
                0.016
            )
            handle_collisions_gpu[blocks, self.threads_per_block](
                cuda.to_device(dummy_positions),
                cuda.to_device(dummy_velocities),
                dummy_size
            )
        except Exception as e:
            print(f"CUDA device not available, using CPU: {e}")
            self.use_gpu = False
    
    def _handle_boundaries_vectorized(self, system):
        """Optimized boundary handling with vectorized operations"""
        active_slice = slice(system.active_particles)
        positions = system.positions[active_slice]
        velocities = system.velocities[active_slice]
        
        # Define boundaries with proper offsets
        left_boundary = PARTICLE_RADIUS * 2
        right_boundary = WINDOW_WIDTH - PARTICLE_RADIUS * 2
        floor_level = SIMULATION_HEIGHT - FLOOR_BUFFER - PARTICLE_RADIUS
        
        # Handle horizontal boundaries
        x_left_violation = positions[:, 0] < left_boundary
        x_right_violation = positions[:, 0] > right_boundary
        
        # Fix positions and reflect velocities
        positions[x_left_violation, 0] = left_boundary
        positions[x_right_violation, 0] = right_boundary
        velocities[x_left_violation, 0] *= -COLLISION_RESPONSE
        velocities[x_right_violation, 0] *= -COLLISION_RESPONSE
        
        # Handle vertical boundaries (floor)
        y_floor_violation = positions[:, 1] > floor_level
        positions[y_floor_violation, 1] = floor_level
        
        # Apply floor collision response with friction
        floor_collision = y_floor_violation & (velocities[:, 1] > 0)
        if np.any(floor_collision):
            velocities[floor_collision, 1] *= -COLLISION_RESPONSE
            velocities[floor_collision, 0] *= 0.95  # Increased friction
        
        # Stop very slow particles on the floor
        floor_contact = positions[:, 1] >= floor_level - 0.1
        slow_velocity = np.abs(velocities[:, 1]) < MIN_BOUNCE_VELOCITY
        stop_mask = floor_contact & slow_velocity
        velocities[stop_mask, 1] = 0
        positions[stop_mask, 1] = floor_level
    
    @profile_function(threshold_ms=1.0)
    def update(self, system, delta_time):
        """Update the physics simulation"""
        if system.active_particles == 0:
            return
            
        # Use simpler physics for small particle counts
        if system.active_particles < self.gpu_batch_threshold:
            self._update_cpu_simple(system, delta_time)
            return
            
        # Adaptive timestep based on particle density
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
        """Optimized GPU implementation"""
        active_slice = slice(system.active_particles)
        
        # Calculate proper grid size
        threads_per_block = self.threads_per_block
        num_particles = system.active_particles
        blocks = (num_particles + threads_per_block - 1) // threads_per_block
        
        if blocks == 0:
            self._update_cpu(system, dt)
            return
        
        # Reuse GPU memory when possible
        if not hasattr(self, 'positions_gpu') or self.positions_gpu.size != system.positions[active_slice].size:
            self.positions_gpu = cuda.to_device(system.positions[active_slice])
            self.velocities_gpu = cuda.to_device(system.velocities[active_slice])
        else:
            self.positions_gpu.copy_to_device(system.positions[active_slice])
            self.velocities_gpu.copy_to_device(system.velocities[active_slice])
        
        # Run simulation steps
        update_velocities_gpu[blocks, threads_per_block](self.velocities_gpu, dt)
        update_positions_gpu[blocks, threads_per_block](self.positions_gpu, self.velocities_gpu, dt)
        handle_collisions_gpu[blocks, threads_per_block](self.positions_gpu, self.velocities_gpu, system.active_particles)
        
        # Copy results back
        self.positions_gpu.copy_to_host(system.positions[active_slice])
        self.velocities_gpu.copy_to_host(system.velocities[active_slice])
        
        # Handle boundaries
        self._handle_boundaries_vectorized(system)

    def _update_cpu_simple(self, system, dt):
        """Simplified CPU physics for small particle counts"""
        active_slice = slice(system.active_particles)
        
        # Basic velocity and position updates
        system.velocities[active_slice, 1] += GRAVITY * dt
        system.velocities[active_slice] = np.clip(
            system.velocities[active_slice],
            -MAX_VELOCITY, MAX_VELOCITY
        )
        
        system.positions[active_slice] += system.velocities[active_slice] * dt
        
        # Simple boundary checks
        self._handle_boundaries_simple(system)

    def _handle_boundaries_simple(self, system):
        """Simplified boundary handling for small particle counts"""
        active_slice = slice(system.active_particles)
        positions = system.positions[active_slice]
        velocities = system.velocities[active_slice]
        
        # Floor collision
        floor_level = SIMULATION_HEIGHT - FLOOR_BUFFER - PARTICLE_RADIUS
        floor_collision = positions[:, 1] > floor_level
        positions[floor_collision, 1] = floor_level
        velocities[floor_collision, 1] *= -COLLISION_RESPONSE
        
        # Wall collisions
        left_wall = positions[:, 0] < PARTICLE_RADIUS * 2
        right_wall = positions[:, 0] > WINDOW_WIDTH - PARTICLE_RADIUS * 2
        
        positions[left_wall, 0] = PARTICLE_RADIUS * 2
        positions[right_wall, 0] = WINDOW_WIDTH - PARTICLE_RADIUS * 2
        
        velocities[left_wall, 0] *= -COLLISION_RESPONSE
        velocities[right_wall, 0] *= -COLLISION_RESPONSE

    def _update_cpu(self, system, delta_time):
        """Optimized CPU implementation with SIMD operations"""
        active_slice = slice(system.active_particles)
        
        # Use pre-allocated arrays for better performance
        if not hasattr(self, '_predicted_positions') or self._predicted_positions.shape != system.positions[active_slice].shape:
            self._predicted_positions = np.empty_like(system.positions[active_slice])
        
        # Vectorized velocity update with gravity
        system.velocities[active_slice, 1] = np.clip(
            system.velocities[active_slice, 1] + GRAVITY * delta_time,
            -MAX_VELOCITY, MAX_VELOCITY
        )
        
        # Predict positions for collision detection
        np.add(system.positions[active_slice], 
               system.velocities[active_slice] * delta_time, 
               out=self._predicted_positions)
        
        # Efficient boundary handling
        floor_level = SIMULATION_HEIGHT - FLOOR_BUFFER - PARTICLE_RADIUS
        floor_collision = self._predicted_positions[:, 1] >= floor_level
        
        if np.any(floor_collision):
            self._predicted_positions[floor_collision, 1] = floor_level
            system.velocities[active_slice][floor_collision, 1] *= -COLLISION_RESPONSE
            # Add friction
            system.velocities[active_slice][floor_collision, 0] *= 0.98
        
        # Update positions
        system.positions[active_slice] = self._predicted_positions
        
        # Handle remaining boundaries and collisions
        self._handle_boundaries_vectorized(system)
        self._handle_collisions_vectorized(system)

    def _handle_collisions_vectorized(self, system):
        """Optimized collision handling with spatial partitioning"""
        active_slice = slice(system.active_particles)
        positions = system.positions[active_slice]
        velocities = system.velocities[active_slice]
        
        # Check all pairs of particles (temporary solution for debugging)
        for i in range(system.active_particles):
            for j in range(i + 1, system.active_particles):
                diff = positions[i] - positions[j]
                dist_sq = np.sum(diff * diff)
                
                # Increase collision detection radius for testing
                collision_radius = PARTICLE_RADIUS * 12.0
                if dist_sq < (collision_radius) ** 2 and dist_sq > 0:
                    dist = np.sqrt(dist_sq)
                    self._handle_collision(system, i, j, dist)

    def _handle_collision(self, system, i, j, dist):
        """Handle collision between two particles"""
        # Only log if debug mode is enabled
        if system.debug_mode:
            elem1 = system.chemical_properties[i].element_data.id
            elem2 = system.chemical_properties[j].element_data.id
            print(f"Collision between {elem1} and {elem2} at distance {dist:.2f}")