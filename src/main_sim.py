import sys
import pygame
from pygame.locals import *
from pygame import Surface
from pygame import Color

from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    BLACK, WHITE, ELEMENT_COLORS
)
from physics.particle_data import ParticleData
from physics.particles_manager import create_partices, render_particles, render_grid
from physics.simulation import solve

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
        self.particles_1 = ParticleData(Color(255, 255, 0))
        
        # Mouse state
        self.mouse_down = False
        
        # Store current delta time
        self.current_delta = 0.0
        
        # Element selection state
        self.selected_element = 'H'  # Default to Hydrogen
        self.element_font = pygame.font.Font(None, 24)
        
        # UI dimensions and layout
        self.ui_padding = 10
        self.tab_height = 40
        self.tab_width = 60
        self.tab_padding = 5
        self.sidebar_width = 200
        self.elements = ['H', 'O', 'N', 'C']
        
        # Create UI surfaces
        self.sidebar = Surface((self.sidebar_width, WINDOW_HEIGHT))
        self.element_tabs = Surface((WINDOW_WIDTH - self.sidebar_width, self.tab_height))
        self.simulation_area = Surface((WINDOW_WIDTH - self.sidebar_width, 
                                      WINDOW_HEIGHT - self.tab_height))
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = pygame.mouse.get_pos()
                    # Adjust mouse position for element tabs
                    if (mouse_pos[0] > self.sidebar_width and 
                        mouse_pos[1] < self.tab_height):
                        adjusted_x = mouse_pos[0] - self.sidebar_width
                        tab_idx = adjusted_x // (self.tab_width + self.tab_padding)
                        if tab_idx < len(self.elements):
                            self.selected_element = self.elements[tab_idx]
                    # Only allow particle creation in simulation area
                    elif (mouse_pos[0] > self.sidebar_width and 
                          mouse_pos[1] > self.tab_height):
                        self.mouse_down = True
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
        return True
    
    def update(self):
        self.current_delta = self.clock.get_time() / 1000.0
        
        if self.mouse_down:
            mouse_pos = pygame.mouse.get_pos()
            # Adjust mouse position relative to simulation area
            adjusted_pos = (
                mouse_pos[0] - self.sidebar_width,
                mouse_pos[1] - self.tab_height
            )
            create_partices(self.particles_1, adjusted_pos)
        
        solve([self.particles_1])
            
    def _draw_sidebar(self):
        """Draw the sidebar with stats and info"""
        self.sidebar.fill(BLACK)
        
        # Draw stats
        y_offset = self.ui_padding
        stats = [
            f'FPS: {int(self.clock.get_fps())}',
            f'Particles: {666}',
            f'Selected: {self.selected_element}'
        ]
        
        for stat in stats:
            text = self.font.render(stat, True, WHITE)
            self.sidebar.blit(text, (self.ui_padding, y_offset))
            y_offset += 40
        
        # Draw sidebar border
        pygame.draw.line(self.sidebar, WHITE, 
                        (self.sidebar_width-1, 0), 
                        (self.sidebar_width-1, WINDOW_HEIGHT), 1)
    
    def _draw_element_tabs(self):
        """Draw the element selection tabs"""
        self.element_tabs.fill(BLACK)
        
        for i, element in enumerate(self.elements):
            x = i * (self.tab_width + self.tab_padding)
            tab_rect = pygame.Rect(x, 0, self.tab_width, self.tab_height-1)
            color = ELEMENT_COLORS[element]
            
            if element == self.selected_element:
                pygame.draw.rect(self.element_tabs, color, tab_rect)
                text_color = BLACK
            else:
                pygame.draw.rect(self.element_tabs, color, tab_rect, 2)
                text_color = color
            
            text = self.element_font.render(element, True, text_color)
            text_rect = text.get_rect(center=(x + self.tab_width/2, self.tab_height/2))
            self.element_tabs.blit(text, text_rect)
        
        # Draw bottom border
        pygame.draw.line(self.element_tabs, WHITE, 
                        (0, self.tab_height-1), 
                        (WINDOW_WIDTH - self.sidebar_width, self.tab_height-1), 1)
    
    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw sidebar
        self._draw_sidebar()
        self.screen.blit(self.sidebar, (0, 0))
        
        # Draw element tabs
        self._draw_element_tabs()
        self.screen.blit(self.element_tabs, (self.sidebar_width, 0))
        
        # Draw simulation area (particles)
        self.simulation_area.fill(BLACK)
        render_particles(self.simulation_area, self.particles_1)
        render_grid(self.simulation_area, self.particles_1)
        self.screen.blit(self.simulation_area, 
                        (self.sidebar_width, self.tab_height))
        
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