import pygame
import sys
from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, BLACK, WHITE,
    MIN_PARTICLES, MAX_PARTICLES
)
from physics.particle_system import ParticleSystem

class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2D Chemistry Simulation")
        self.clock = pygame.time.Clock()
        self.particle_system = ParticleSystem()
        self.running = True
        self.mouse_held = False
        self.font = pygame.font.Font(None, 36)
        self.delta_time = 0
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_held = True
                self.particle_system.create_particle_burst(pygame.mouse.get_pos(), self.delta_time)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_held = False
            elif event.type == pygame.MOUSEWHEEL:
                self.particle_system.particles_per_frame = max(MIN_PARTICLES,
                    min(MAX_PARTICLES,
                        self.particle_system.particles_per_frame + event.y))
        
        if self.mouse_held:
            self.particle_system.create_particle_burst(pygame.mouse.get_pos(), self.delta_time)
    
    def update(self):
        self.delta_time = self.clock.get_time() / 1000.0
        self.particle_system.update(self.delta_time)
    
    def draw(self):
        self.screen.fill(BLACK)
        
        self.particle_system.draw(self.screen)
        
        metrics = [
            f'FPS: {self.clock.get_fps():.1f}',
            f'Delta Time: {self.delta_time*1000:.2f}ms',
            f'Particles: {len(self.particle_system.particles)}'
        ]
        
        for i, text in enumerate(metrics):
            surface = self.font.render(text, True, WHITE)
            self.screen.blit(surface, (10, 10 + i * 40))
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick()

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()
    pygame.quit()
    sys.exit() 