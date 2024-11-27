import numpy as np
import pygame

from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    PARTICLE_SPREAD, GRAVITY, COLLISION_RESPONSE,
    FIXED_TIMESTEP, VELOCITY_SCALE, WHITE, ELEMENT_COLORS
)

class ParticleSystem:
    def __init__(self):
        self.max_particles = 20000
        self._initialize_arrays()
        self._initialize_grid()
        self.particles_per_frame = PARTICLE_SPREAD
        self.element_types = np.zeros(self.max_particles, dtype='U1')

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
        
    def create_particle_burst(self, pos, _, element_type='H'):
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
        """Handle collision between two particles"""
        vertical_dist = self.positions[current_idx, 1] - self.positions[other_idx, 1]
        
        # Adjust vertical velocities based on relative positions
        if vertical_dist < 0:  # Current particle is above
            self.velocities[current_idx, 1] = max(1.0, self.velocities[current_idx, 1])
        elif vertical_dist > 0:  # Current particle is below
            self.velocities[other_idx, 1] = max(1.0, self.velocities[other_idx, 1])
        
        # Add horizontal scatter
        self.velocities[current_idx, 0] += np.random.uniform(-0.5, 0.5)
        self.velocities[other_idx, 0] += np.random.uniform(-0.5, 0.5)
        
        # Prevent overlap
        self.positions[current_idx, 1] -= 0.5
        self.positions[other_idx, 1] += 0.5

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
        """Draw particles to the screen"""
        for i in range(self.active_particles):
            pos = self.positions[i].astype(np.int32)
            color = ELEMENT_COLORS[self.element_types[i]]
            screen.set_at(pos, color)

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