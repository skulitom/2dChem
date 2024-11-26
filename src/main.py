import sys
import pygame
from pygame.locals import *
from pygame import Surface

from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    BLACK, WHITE
)
from physics.particle_system import ParticleSystem

class Simulation:
    def __init__(self):
        pygame.init()
        
        # Set up display
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Particle System")
        
        # Set up clock
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize particle system
        self.particle_system = ParticleSystem()
        
        # Mouse state
        self.mouse_down = False
        
        # Store current delta time
        self.current_delta = 0.0
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_down = True
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:  # Left click
                    self.mouse_down = False
        return True
    
    def update(self):
        self.current_delta = self.clock.get_time() / 1000.0  # Convert to seconds
        
        # Create particles at mouse position if mouse button is held
        if self.mouse_down:
            mouse_pos = pygame.mouse.get_pos()
            self.particle_system.create_particle_burst(mouse_pos, self.current_delta)
        
        # Update particle system
        self.particle_system.update(self.current_delta)
    
    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw particles
        self.particle_system.draw(self.screen)
        
        # Draw FPS, particle count, and delta time
        fps_text = self.font.render(f'FPS: {int(self.clock.get_fps())}', True, WHITE)
        particle_text = self.font.render(
            f'Particles: {self.particle_system.active_particles}',
            True, WHITE
        )
        delta_text = self.font.render(
            f'Delta: {self.current_delta:.4f}s',
            True, WHITE
        )
        
        self.screen.blit(fps_text, (10, 10))
        self.screen.blit(particle_text, (10, 50))
        self.screen.blit(delta_text, (10, 90))
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    simulation = Simulation()
    simulation.run() 