from .core import ParticleSystemCore
from .collision import CollisionHandler
from .renderer import ParticleRenderer
from .physics import PhysicsHandler
from audio.sound_manager import SoundManager

class ParticleSystem(ParticleSystemCore):
    def __init__(self):
        super().__init__()
        self.physics = PhysicsHandler()
        self.collision = CollisionHandler()
        self.renderer = ParticleRenderer()
        self.sound_manager = SoundManager()
        self.debug_mode = False
        self.time = 0.0

    def update(self, delta_time):
        if self.active_particles > 0:
            self.physics.update(self, delta_time)
            self.time += delta_time

    def draw(self, screen):
        self.renderer.draw(self, screen) 

    def create_particle_burst(self, pos, element_type='H', burst_size=None, spread=20.0, speed=2.0):
        """Create a burst of particles with sound"""
        # Create particles (existing functionality)
        super().create_particle_burst(pos, element_type=element_type, burst_size=burst_size)
        
        # Play creation sound
        self.sound_manager.play_creation_sound(element_type) 