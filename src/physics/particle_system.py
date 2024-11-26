from collections import defaultdict
import random
from pygame import Surface, SRCALPHA
import pygame
from pygame.math import Vector2

from core.constants import (
    CELL_SIZE, PARTICLE_RADIUS, WHITE,
    COLLISION_DAMPING, WINDOW_WIDTH, WINDOW_HEIGHT,
    PARTICLE_SPREAD
)
from physics.particle import Particle

class ParticleSystem:
    def __init__(self):
        self.particles = []
        self.grid = defaultdict(list)
        self.particles_per_frame = PARTICLE_SPREAD
        self.neighbor_offsets = [(0,1), (1,0), (1,1), (-1,1)]
        
        # Create particle template surface
        self.particle_surface = Surface((PARTICLE_RADIUS * 2 + 1, PARTICLE_RADIUS * 2 + 1), SRCALPHA)
        pygame.draw.circle(self.particle_surface, WHITE, 
                         (PARTICLE_RADIUS, PARTICLE_RADIUS), PARTICLE_RADIUS)
    
    def create_particle_burst(self, pos, delta_time):
        new_particles = []
        base_pos = Vector2(pos)
        
        # Use a fixed number of particles per burst instead of scaling with delta_time
        for _ in range(self.particles_per_frame):
            offset = Vector2(
                random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD),
                random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD)
            )
            new_pos = base_pos + offset
            new_pos.x = max(PARTICLE_RADIUS, min(new_pos.x, WINDOW_WIDTH - PARTICLE_RADIUS))
            new_pos.y = max(PARTICLE_RADIUS, min(new_pos.y, WINDOW_HEIGHT - PARTICLE_RADIUS))
            
            new_particle = Particle(new_pos)
            new_particles.append(new_particle)
        
        self.particles.extend(new_particles)

    def get_grid_coords(self, particle):
        return (int(particle.position.x // CELL_SIZE),
                int(particle.position.y // CELL_SIZE))
    
    def update(self, delta_time):
        self.grid.clear()
        
        for particle in self.particles:
            particle.update(delta_time)
            grid_pos = self.get_grid_coords(particle)
            self.grid[grid_pos].append(particle)
        
        processed = set()
        grid_items = list(self.grid.items())
        
        for grid_pos, particles in grid_items:
            x, y = grid_pos
            
            for i, particle1 in enumerate(particles):
                for particle2 in particles[i + 1:]:
                    pair_id = (id(particle1), id(particle2))
                    if pair_id not in processed:
                        self.check_collision(particle1, particle2, delta_time)
                        processed.add(pair_id)
                
                for dx, dy in self.neighbor_offsets:
                    neighbor_pos = (x + dx, y + dy)
                    if neighbor_pos in self.grid:
                        for particle2 in self.grid[neighbor_pos]:
                            pair_id = (id(particle1), id(particle2))
                            if pair_id not in processed:
                                self.check_collision(particle1, particle2, delta_time)
                                processed.add(pair_id)
    
    @staticmethod
    def check_collision(p1, p2, delta_time):
        dx = p1.position.x - p2.position.x
        dy = p1.position.y - p2.position.y
        dist_sq = dx * dx + dy * dy
        
        if dist_sq < 4 * PARTICLE_RADIUS * PARTICLE_RADIUS:
            dist = (dist_sq ** 0.5) or 1
            
            nx = dx / dist
            ny = dy / dist
            
            rvx = p1.velocity.x - p2.velocity.x
            rvy = p1.velocity.y - p2.velocity.y
            
            # Scale impulse with delta time
            impulse = -(rvx * nx + rvy * ny) * COLLISION_DAMPING * delta_time
            
            p1.velocity.x += nx * impulse
            p1.velocity.y += ny * impulse
            p2.velocity.x -= nx * impulse
            p2.velocity.y -= ny * impulse
            
            # Scale position correction with delta time
            overlap = (2 * PARTICLE_RADIUS - dist) / 2 * delta_time
            p1.position.x += nx * overlap
            p1.position.y += ny * overlap
            p2.position.x -= nx * overlap
            p2.position.y -= ny * overlap
    
    def draw(self, screen):
        for particle in self.particles:
            screen.blit(self.particle_surface, 
                       (int(particle.position.x - PARTICLE_RADIUS),
                        int(particle.position.y - PARTICLE_RADIUS))) 