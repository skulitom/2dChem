import numpy as np
from core.constants import GRAVITY, WINDOW_HEIGHT, COLLISION_RESPONSE, FIXED_TIMESTEP
from .collision import CollisionHandler

class PhysicsHandler:
    def update(self, system, delta_time):
        if system.active_particles > 0:
            self._fixed_update(system, FIXED_TIMESTEP)
    
    def _fixed_update(self, system, dt):
        if system.active_particles == 0:
            return
        
        self._update_velocities(system, dt)
        self._update_positions(system, dt)
        self._handle_boundaries(system)
        self._handle_collisions(system)

    def _update_velocities(self, system, dt):
        system.velocities[:system.active_particles] += np.array([0, GRAVITY * dt])
        
        mask = (system.positions[:system.active_particles, 1] < (WINDOW_HEIGHT - 1)) & \
               (system.velocities[:system.active_particles, 1] < 0.5)
        system.velocities[:system.active_particles, 1][mask] = 0.5

    def _update_positions(self, system, dt):
        system.positions[:system.active_particles] += system.velocities[:system.active_particles] * dt

    def _handle_boundaries(self, system):
        active_mask = np.arange(system.active_particles)
        
        bottom_mask = active_mask[system.positions[active_mask, 1] >= system.grid_size_y - 1]
        system.positions[bottom_mask, 1] = system.grid_size_y - 1
        system.velocities[bottom_mask, 1] *= -COLLISION_RESPONSE
        system.velocities[bottom_mask, 0] += np.random.uniform(-0.3, 0.3, len(bottom_mask))
        
        left_mask = active_mask[system.positions[active_mask, 0] < 0]
        system.positions[left_mask, 0] = 0
        system.velocities[left_mask, 0] = abs(system.velocities[left_mask, 0]) * COLLISION_RESPONSE
        
        right_mask = active_mask[system.positions[active_mask, 0] >= system.grid_size_x - 1]
        system.positions[right_mask, 0] = system.grid_size_x - 1
        system.velocities[right_mask, 0] = -abs(system.velocities[right_mask, 0]) * COLLISION_RESPONSE

    def _handle_collisions(self, system):
        if system.active_particles < 2:
            return
        
        system.grid.fill(0)
        active_positions = system.positions[:system.active_particles]
        
        pixel_positions = np.clip(
            active_positions.astype(np.int32),
            [0, 0],
            [system.grid_size_x - 1, system.grid_size_y - 1]
        )
        
        grid_indices = pixel_positions[:, 1] * system.grid_size_x + pixel_positions[:, 0]
        unique_indices, counts = np.unique(grid_indices, return_counts=True)
        collision_cells = unique_indices[counts > 1]
        
        for cell_idx in collision_cells:
            y, x = divmod(cell_idx, system.grid_size_x)
            particle_indices = np.where((pixel_positions[:, 0] == x) & 
                                     (pixel_positions[:, 1] == y))[0]
            
            for i in range(len(particle_indices)):
                for j in range(i + 1, len(particle_indices)):
                    CollisionHandler.handle_particle_collision(
                        system, 
                        particle_indices[i], 
                        particle_indices[j]
                    ) 