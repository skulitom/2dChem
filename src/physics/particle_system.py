import numpy as np
import pygame

from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    PARTICLE_SPREAD, GRAVITY, COLLISION_RESPONSE,
    FIXED_TIMESTEP, VELOCITY_SCALE, WHITE, ELEMENT_COLORS
)
from physics.chemical_particle import ChemicalParticle

class ParticleSystem:
    def __init__(self):
        self.max_particles = 20000
        self._initialize_arrays()
        self._initialize_grid()
        self.particles_per_frame = PARTICLE_SPREAD
        self.element_types = np.zeros(self.max_particles, dtype='U1')
        self.chemical_properties = {}  # Dictionary to store ChemicalParticle instances

    def _initialize_arrays(self):
        """Initialize position and velocity arrays"""
        self.positions = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.active_particles = 0

    def _initialize_grid(self):
        """Initialize spatial partitioning grid"""
        self.grid_size_x = WINDOW_WIDTH - 200
        self.grid_size_y = WINDOW_HEIGHT - 40
        self.grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=np.int32)
        
    def create_particle_burst(self, pos, delta_time, element_type='H'):
        """Create a burst of particles at the given position"""
        new_count = min(self.particles_per_frame, self.max_particles - self.active_particles)
        if new_count <= 0:
            return
        
        # Generate new particle positions and velocities
        offsets = np.random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD, (new_count, 2))
        new_positions = np.tile(pos, (new_count, 1)) + offsets
        new_velocities = np.random.uniform(-0.5, 0.5, (new_count, 2)) * VELOCITY_SCALE
        
        # Clamp positions to window boundaries
        new_positions[:, 0] = np.clip(new_positions[:, 0], 0, WINDOW_WIDTH - 1)
        new_positions[:, 1] = np.clip(new_positions[:, 1], 0, WINDOW_HEIGHT - 1)
        
        # Update particle arrays
        start_idx = self.active_particles
        end_idx = start_idx + new_count
        self.positions[start_idx:end_idx] = new_positions
        self.velocities[start_idx:end_idx] = new_velocities
        self.active_particles += new_count
        
        # Set element types for new particles
        self.element_types[start_idx:end_idx] = element_type
        
        # Create chemical properties for new particles
        for i in range(start_idx, end_idx):
            self.chemical_properties[i] = ChemicalParticle(element_type, i)

    def _handle_boundaries(self):
        """Handle collisions with window boundaries"""
        active_mask = np.arange(self.active_particles)
        
        # Bottom boundary (adjusted for tab height)
        bottom_mask = active_mask[self.positions[active_mask, 1] >= self.grid_size_y - 1]
        self.positions[bottom_mask, 1] = self.grid_size_y - 1
        self.velocities[bottom_mask, 1] *= -COLLISION_RESPONSE
        self.velocities[bottom_mask, 0] += np.random.uniform(-0.3, 0.3, len(bottom_mask))
        
        # Side boundaries
        left_mask = active_mask[self.positions[active_mask, 0] < 0]
        self.positions[left_mask, 0] = 0
        self.velocities[left_mask, 0] = abs(self.velocities[left_mask, 0]) * COLLISION_RESPONSE
        
        right_mask = active_mask[self.positions[active_mask, 0] >= self.grid_size_x - 1]
        self.positions[right_mask, 0] = self.grid_size_x - 1
        self.velocities[right_mask, 0] = -abs(self.velocities[right_mask, 0]) * COLLISION_RESPONSE

    def _handle_particle_collision(self, current_idx, other_idx):
        """Handle collision between two particles with potential chemical reaction"""
        current_chem = self.chemical_properties[current_idx]
        other_chem = self.chemical_properties[other_idx]
        
        # Calculate kinetic energy and convert to temperature
        rel_velocity = np.linalg.norm(self.velocities[current_idx] - self.velocities[other_idx])
        collision_energy = 0.5 * (current_chem.element_data.mass * rel_velocity ** 2)
        
        # Update particle temperatures based on collision energy
        temp_increase = collision_energy * 0.1  # Scale factor for gameplay
        current_chem.temperature += temp_increase
        other_chem.temperature += temp_increase
        
        # Check for phase changes and decomposition
        self._handle_phase_changes(current_idx)
        self._handle_phase_changes(other_idx)
        
        # Calculate distance between particles
        pos_diff = self.positions[current_idx] - self.positions[other_idx]
        distance = np.linalg.norm(pos_diff)
        
        # Only allow bonding if temperature conditions are suitable
        if (current_chem.temperature < current_chem.element_data.boiling_point and
            other_chem.temperature < other_chem.element_data.boiling_point):
            if current_chem.try_form_bond(other_chem, distance):
                self._handle_bonded_particles(current_idx, other_idx)
        else:
            # Regular collision handling
            if distance > 0:
                # Calculate collision response
                direction = pos_diff / distance
                relative_velocity = self.velocities[current_idx] - self.velocities[other_idx]
                
                # Apply collision impulse
                impulse = direction * np.dot(relative_velocity, direction) * COLLISION_RESPONSE
                
                self.velocities[current_idx] -= impulse
                self.velocities[other_idx] += impulse
                
                # Separate particles to prevent overlap
                overlap = 2 - distance  # Assuming radius of 1 for each particle
                if overlap > 0:
                    separation = direction * overlap * 0.5
                    self.positions[current_idx] += separation
                    self.positions[other_idx] -= separation

    def _handle_bonded_particles(self, idx1, idx2):
        """Handle movement of bonded particles"""
        # Maintain bond distance
        pos_diff = self.positions[idx1] - self.positions[idx2]
        distance = np.linalg.norm(pos_diff)
        target_distance = (self.chemical_properties[idx1].element_data.radius +
                         self.chemical_properties[idx2].element_data.radius)
        
        if distance > 0:
            correction = (distance - target_distance) * 0.5
            direction = pos_diff / distance
            self.positions[idx1] -= direction * correction
            self.positions[idx2] += direction * correction
            
            # Average out velocities for bonded particles
            avg_velocity = (self.velocities[idx1] + self.velocities[idx2]) * 0.5
            self.velocities[idx1] = avg_velocity
            self.velocities[idx2] = avg_velocity

    def _handle_collisions(self):
        """Handle collisions between particles using spatial partitioning"""
        if self.active_particles < 2:
            return
            
        self.grid.fill(0)
        pixel_positions = self.positions[:self.active_particles].astype(np.int32)
        pixel_positions[:, 0] = np.clip(pixel_positions[:, 0], 0, WINDOW_WIDTH - 1)
        pixel_positions[:, 1] = np.clip(pixel_positions[:, 1], 0, WINDOW_HEIGHT - 1)
        
        for i in range(self.active_particles):
            px, py = pixel_positions[i]
            if self.grid[px, py] != 0:
                self._handle_particle_collision(i, self.grid[px, py] - 1)
            else:
                self.grid[px, py] = i + 1

    def draw(self, screen):
        """Draw particles with temperature-based coloring"""
        for i in range(self.active_particles):
            pos = self.positions[i].astype(np.int32)
            
            # Convert Pygame Color to RGB array
            element_color = ELEMENT_COLORS[self.element_types[i]]
            base_color = np.array([element_color.r, element_color.g, element_color.b])
            
            # Temperature-based color modification
            temp = self.chemical_properties[i].temperature
            temp_factor = min(1.0, max(0, (temp - 273.15) / 1000))  # Scale temperature effect
            hot_color = np.array([255, 200, 0])  # Yellow-orange for hot particles
            
            # Mix colors based on temperature
            mixed_color = tuple((base_color * (1 - temp_factor) + hot_color * temp_factor).astype(int))
            
            # Draw particle
            pygame.draw.circle(screen, mixed_color, pos, 2)
            
            # Draw bonds
            for bond in self.chemical_properties[i].bonds:
                if bond.particle_id < i:  # Only draw each bond once
                    bond_pos = self.positions[bond.particle_id].astype(np.int32)
                    # Draw bond with strength indicator
                    bond_color = (200, 200, 200)  # Default gray
                    if bond.bond_type == 'covalent':
                        # Brighter color for stronger bonds
                        intensity = int(155 + bond.strength * 40)
                        bond_color = (intensity, intensity, intensity)
                    elif bond.bond_type == 'ionic':
                        # Blueish for ionic bonds
                        bond_color = (100, 100, 255)
                    pygame.draw.line(screen, bond_color, pos, bond_pos, 1)

    def update(self, delta_time):
        """Update particle system"""
        if self.active_particles > 0:
            self._fixed_update(FIXED_TIMESTEP)
    
    def _fixed_update(self, dt):
        """Physics update with fixed timestep"""
        if self.active_particles == 0:
            return
            
        # Update velocities and apply gravity
        self.velocities[:self.active_particles, 1] += GRAVITY * dt
        
        # Ensure minimum downward velocity
        not_at_bottom = self.positions[:self.active_particles, 1] < (WINDOW_HEIGHT - 1)
        falling_too_slow = (self.velocities[:self.active_particles, 1] < 0.5) & not_at_bottom
        self.velocities[:self.active_particles, 1][falling_too_slow] = 0.5
        
        # Update positions
        self.positions[:self.active_particles] += self.velocities[:self.active_particles] * dt
        
        self._handle_boundaries()
        self._handle_collisions()

    def _handle_phase_changes(self, idx):
        """Handle particle phase changes based on temperature"""
        chem = self.chemical_properties[idx]
        
        # Phase change affects particle behavior
        if chem.temperature >= chem.element_data.boiling_point:
            self.velocities[idx] += np.random.uniform(-1, 1, 2) * 2  # More energetic motion
            chem.break_all_bonds()  # Break bonds when boiling
        elif chem.temperature <= chem.element_data.melting_point:
            self.velocities[idx] *= 0.8  # Slower motion when freezing