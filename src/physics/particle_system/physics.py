import numpy as np
from core.constants import GRAVITY, WINDOW_HEIGHT, COLLISION_RESPONSE, FIXED_TIMESTEP
from .collision import CollisionHandler
from utils.profiler import profile_function

class PhysicsHandler:
    @profile_function(threshold_ms=1.0)
    def update(self, system, delta_time):
        if system.active_particles > 0:
            system.active_particles = min(system.active_particles, len(system.positions))
            self._fixed_update(system, FIXED_TIMESTEP)
    
    @profile_function(threshold_ms=0.5)
    def _fixed_update(self, system, dt):
        if system.active_particles == 0:
            return
        
        self._update_velocities(system, dt)
        self._update_positions(system, dt)
        self._handle_boundaries(system)
        self._handle_collisions(system)

    @profile_function(threshold_ms=0.5)
    def _update_velocities(self, system, dt):
        active_mask = np.arange(len(system.positions)) < system.active_particles
        system.velocities[active_mask] += np.array([0, GRAVITY * dt])
        
        np.clip(system.velocities[active_mask, 1], -20, 20, out=system.velocities[active_mask, 1])

    @profile_function(threshold_ms=0.5)
    def _update_positions(self, system, dt):
        system.positions[:system.active_particles] += system.velocities[:system.active_particles] * dt

    @profile_function(threshold_ms=0.5)
    def _handle_boundaries(self, system):
        # Get active particles positions and velocities as views
        positions = system.positions[:system.active_particles]
        velocities = system.velocities[:system.active_particles]
        
        # Add bounds checking for y-coordinate
        np.clip(positions[:, 1], 0, system.grid_size_y - 2, out=positions[:, 1])
        
        # Process all boundaries at once using masks
        bottom_mask = positions[:, 1] >= system.grid_size_y - 4  # Increased buffer zone
        left_mask = positions[:, 0] < 2
        right_mask = positions[:, 0] >= system.grid_size_x - 2
        
        # Apply boundary conditions using vectorized operations
        positions[bottom_mask, 1] = system.grid_size_y - 4  # Keep particles further from edge
        positions[left_mask, 0] = 2
        positions[right_mask, 0] = system.grid_size_x - 2
        
        # Update velocities with stronger collision response for floor
        velocities[bottom_mask, 1] *= -COLLISION_RESPONSE * 1.2  # Increased bounce response
        velocities[left_mask, 0] = np.abs(velocities[left_mask, 0]) * COLLISION_RESPONSE
        velocities[right_mask, 0] = -np.abs(velocities[right_mask, 0]) * COLLISION_RESPONSE
        
        # Ensure particles at bottom have enough upward velocity
        floor_particles = bottom_mask & (np.abs(velocities[:, 1]) < 0.5)
        velocities[floor_particles, 1] = 0.5
        
        # Add random horizontal velocity to bottom collisions to prevent stacking
        if np.any(bottom_mask):
            velocities[bottom_mask, 0] += np.random.uniform(-0.5, 0.5, np.sum(bottom_mask))

    @profile_function(threshold_ms=0.5)
    def _handle_collisions(self, system):
        if system.active_particles < 2:
            return
        
        # Create spatial grid for collision detection
        grid_size = 4  # Adjust based on typical particle size and velocity
        positions = system.positions[:system.active_particles]
        
        # Calculate grid indices more efficiently
        grid_x = np.clip(positions[:, 0] // grid_size, 0, system.grid_size_x // grid_size - 1).astype(np.int32)
        grid_y = np.clip(positions[:, 1] // grid_size, 0, system.grid_size_y // grid_size - 1).astype(np.int32)
        
        # Create grid using numpy operations
        grid_indices = grid_y * (system.grid_size_x // grid_size) + grid_x
        unique_cells, cell_counts = np.unique(grid_indices, return_counts=True)
        cells_with_multiple = unique_cells[cell_counts > 1]
        
        # Process only cells with potential collisions
        for cell_idx in cells_with_multiple:
            cell_particles = np.where(grid_indices == cell_idx)[0]
            if len(cell_particles) > 1:
                # Get positions for current cell particles
                cell_positions = positions[cell_particles]
                
                # Calculate all pairwise distances efficiently
                diffs = cell_positions[:, np.newaxis] - cell_positions
                distances = np.linalg.norm(diffs, axis=2)
                
                # Find close pairs
                close_pairs = np.where((distances < 3.0) & (distances > 0))
                
                # Process only unique pairs
                for i, j in zip(*close_pairs):
                    if i < j:  # Avoid duplicate pairs
                        idx1, idx2 = cell_particles[i], cell_particles[j]
                        CollisionHandler.handle_particle_collision(system, idx1, idx2) 