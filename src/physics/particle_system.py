import numpy as np
from collections import defaultdict
import pygame

from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    PARTICLE_SPREAD, GRAVITY, COLLISION_RESPONSE,
    FIXED_TIMESTEP, VELOCITY_SCALE, WHITE
)

class ParticleSystem:
    def __init__(self):
        self.max_particles = 20000  # Can handle more particles now that they're smaller
        
        # Position and velocity arrays
        self.positions = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.active_particles = 0
        
        # Grid system - now using 1x1 cell size for pixel-perfect collisions
        self.grid_size_x = WINDOW_WIDTH
        self.grid_size_y = WINDOW_HEIGHT
        self.grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=np.int32)
        
        self.particles_per_frame = PARTICLE_SPREAD
        
    def create_particle_burst(self, pos, _):
        new_count = min(self.particles_per_frame, self.max_particles - self.active_particles)
        if new_count <= 0:
            return
        
        offsets = np.random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD, (new_count, 2))
        new_positions = np.tile(pos, (new_count, 1)) + offsets
        
        # Clamp positions to window boundaries (now using 0 to WIDTH-1)
        new_positions[:, 0] = np.clip(new_positions[:, 0], 0, WINDOW_WIDTH - 1)
        new_positions[:, 1] = np.clip(new_positions[:, 1], 0, WINDOW_HEIGHT - 1)
        
        new_velocities = np.random.uniform(-0.5, 0.5, (new_count, 2)) * VELOCITY_SCALE
        
        start_idx = self.active_particles
        end_idx = start_idx + new_count
        
        self.positions[start_idx:end_idx] = new_positions
        self.velocities[start_idx:end_idx] = new_velocities
        self.active_particles += new_count

    def _handle_boundaries(self):
        active_mask = np.arange(self.active_particles)
        
        # Bottom boundary - stronger bounce response
        bottom_mask = active_mask[self.positions[active_mask, 1] >= WINDOW_HEIGHT - 1]
        self.positions[bottom_mask, 1] = WINDOW_HEIGHT - 1
        self.velocities[bottom_mask, 1] *= -COLLISION_RESPONSE
        # Add some random horizontal velocity on bottom collision
        self.velocities[bottom_mask, 0] += np.random.uniform(-0.3, 0.3, len(bottom_mask))
        
        # Left boundary
        left_mask = active_mask[self.positions[active_mask, 0] < 0]
        self.positions[left_mask, 0] = 0
        self.velocities[left_mask, 0] = abs(self.velocities[left_mask, 0]) * COLLISION_RESPONSE
        
        # Right boundary
        right_mask = active_mask[self.positions[active_mask, 0] >= WINDOW_WIDTH - 1]
        self.positions[right_mask, 0] = WINDOW_WIDTH - 1
        self.velocities[right_mask, 0] = -abs(self.velocities[right_mask, 0]) * COLLISION_RESPONSE

    def _handle_collisions(self):
        if self.active_particles < 2:
            return
            
        # Reset grid
        self.grid.fill(0)
        
        # Round positions to integers for grid placement
        pixel_positions = self.positions[:self.active_particles].astype(np.int32)
        
        # Ensure positions are within bounds
        pixel_positions[:, 0] = np.clip(pixel_positions[:, 0], 0, WINDOW_WIDTH - 1)
        pixel_positions[:, 1] = np.clip(pixel_positions[:, 1], 0, WINDOW_HEIGHT - 1)
        
        # Find collisions using the grid
        for i in range(self.active_particles):
            px, py = pixel_positions[i]
            
            # If cell is already occupied, we have a collision
            if self.grid[px, py] != 0:
                # Get the index of the particle we're colliding with
                other_idx = self.grid[px, py] - 1
                
                # Calculate vertical distance between particles
                vertical_dist = self.positions[i, 1] - self.positions[other_idx, 1]
                
                # If upper particle, ensure it keeps falling
                if vertical_dist < 0:  # Current particle is above
                    self.velocities[i, 1] = max(1.0, self.velocities[i, 1])
                elif vertical_dist > 0:  # Current particle is below
                    self.velocities[other_idx, 1] = max(1.0, self.velocities[other_idx, 1])
                
                # Add horizontal scatter to prevent perfect stacking
                self.velocities[i, 0] += np.random.uniform(-0.5, 0.5)
                self.velocities[other_idx, 0] += np.random.uniform(-0.5, 0.5)
                
                # Move particles apart slightly to prevent overlap
                self.positions[i, 1] -= 0.5
                self.positions[other_idx, 1] += 0.5
            else:
                # Mark cell as occupied with particle index + 1 (0 means empty)
                self.grid[px, py] = i + 1

    def draw(self, screen):
        # Convert positions to integers for pixel drawing
        pixel_positions = self.positions[:self.active_particles].astype(np.int32)
        
        # Draw all particles as single pixels
        for pos in pixel_positions:
            screen.set_at(pos, WHITE)

    def update(self, delta_time):
        if self.active_particles == 0:
            return
            
        # Update using fixed timestep
        self._fixed_update(FIXED_TIMESTEP)
    
    def _fixed_update(self, dt):
        """Physics update with fixed timestep"""
        if self.active_particles == 0:
            return
            
        # Update velocities (gravity)
        self.velocities[:self.active_particles, 1] += GRAVITY * dt
        
        # Ensure minimum downward velocity for all particles not at bottom
        bottom_threshold = WINDOW_HEIGHT - 1
        not_at_bottom = self.positions[:self.active_particles, 1] < bottom_threshold
        min_fall_speed = 0.5
        
        falling_too_slow = (self.velocities[:self.active_particles, 1] < min_fall_speed) & not_at_bottom
        self.velocities[:self.active_particles, 1][falling_too_slow] = min_fall_speed
        
        # Update positions
        self.positions[:self.active_particles] += self.velocities[:self.active_particles] * dt
        
        # Handle collisions with boundaries
        self._handle_boundaries()
        
        # Handle particle collisions using spatial partitioning
        self._handle_collisions()