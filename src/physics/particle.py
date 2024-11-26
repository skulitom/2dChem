from pygame.math import Vector2
import random
from core.constants import (
    PARTICLE_RADIUS, WINDOW_WIDTH, WINDOW_HEIGHT,
    COLLISION_DAMPING, GRAVITY
)

class Particle:
    __slots__ = ('position', 'velocity')
    
    def __init__(self, pos):
        self.position = Vector2(pos)
        self.velocity = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
    
    def update(self, delta_time):
        # Scale physics with delta time
        self.velocity.y += GRAVITY * delta_time * 60
        
        # Update position with delta time
        self.position += self.velocity * delta_time * 60
        
        # Boundary collision
        if self.position.y >= WINDOW_HEIGHT - PARTICLE_RADIUS:
            self.position.y = WINDOW_HEIGHT - PARTICLE_RADIUS
            self.velocity.y *= -COLLISION_DAMPING
        
        if self.position.x < PARTICLE_RADIUS:
            self.position.x = PARTICLE_RADIUS
            self.velocity.x *= -COLLISION_DAMPING
        elif self.position.x >= WINDOW_WIDTH - PARTICLE_RADIUS:
            self.position.x = WINDOW_WIDTH - PARTICLE_RADIUS
            self.velocity.x *= -COLLISION_DAMPING 