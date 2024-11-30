from .core import ParticleSystemCore
from .collision import CollisionHandler
from .renderer import ParticleRenderer
from .physics import PhysicsHandler

class ParticleSystem(ParticleSystemCore):
    def __init__(self):
        super().__init__()
        self.physics = PhysicsHandler()
        self.collision = CollisionHandler()
        self.renderer = ParticleRenderer()
        self.debug_mode = False

    def update(self, delta_time):
        if self.active_particles > 0:
            self.physics.update(self, delta_time)

    def draw(self, screen):
        self.renderer.draw(self, screen) 