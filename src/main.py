import sys
import pygame
from pygame.locals import *
from pygame import Surface

from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    BLACK, WHITE, ELEMENT_COLORS,
    UI_COLORS,
    SIMULATION_FRAME_X_OFFSET, SIMULATION_FRAME_Y_OFFSET
)
from physics.particle_system import ParticleSystem

class Simulation:
    def __init__(self):
        pygame.init()
        
        # UI configuration
        self._init_ui_config()
        
        # Initialize core components
        self._init_display()
        self._init_state()
        self._init_surfaces()
    
    def _init_ui_config(self):
        """Initialize UI configuration values"""
        self.ui_padding = 10
        self.tab_height = 40
        self.tab_width = 70
        self.tab_padding = 2
        self.sidebar_width = 200
        self.elements = ['H', 'O', 'N', 'C']
        self.element_font = pygame.font.Font(None, 28)
        self.stats_font = pygame.font.Font(None, 24)
        self.corner_radius = 5
    
    def _init_display(self):
        """Initialize display and timing components"""
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Particle System")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
    
    def _init_state(self):
        """Initialize simulation state"""
        self.particle_system = ParticleSystem()
        self.mouse_down = False
        self.current_delta = 0.0
        self.selected_element = 'H'  # Default to Hydrogen
        self.sidebar_width = SIMULATION_FRAME_X_OFFSET
        self.tab_height = SIMULATION_FRAME_Y_OFFSET
        self.particles_per_burst = 5  # Initialize with a default value
    
    def _init_surfaces(self):
        """Initialize UI surfaces"""
        self.sidebar = Surface((SIMULATION_FRAME_X_OFFSET, WINDOW_HEIGHT))
        self.element_tabs = Surface((WINDOW_WIDTH - self.sidebar_width, self.tab_height))
        self.simulation_area = Surface((
            WINDOW_WIDTH - self.sidebar_width,
            WINDOW_HEIGHT - self.tab_height
        ))
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_mouse_down(event)
                elif event.button == 4:  # Mouse wheel up
                    self.particles_per_burst = min(100, self.particles_per_burst + 5)
                elif event.button == 5:  # Mouse wheel down
                    self.particles_per_burst = max(1, self.particles_per_burst - 5)
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                self.mouse_down = False
        return True
    
    def _handle_mouse_down(self, event):
        """Handle mouse down event"""
        mouse_pos = pygame.mouse.get_pos()
        
        if self._is_in_element_tabs(mouse_pos):
            self._handle_tab_selection(mouse_pos)
        elif self._is_in_simulation_area(mouse_pos):
            self.mouse_down = True
    
    def _is_in_element_tabs(self, pos):
        """Check if position is in element tabs area"""
        return (pos[0] > self.sidebar_width and pos[1] < self.tab_height)
    
    def _is_in_simulation_area(self, pos):
        """Check if position is in simulation area"""
        return (pos[0] > self.sidebar_width and pos[1] > self.tab_height)
    
    def _handle_tab_selection(self, mouse_pos):
        """Handle element tab selection"""
        adjusted_x = mouse_pos[0] - self.sidebar_width
        tab_idx = adjusted_x // (self.tab_width + self.tab_padding)
        if tab_idx < len(self.elements):
            self.selected_element = self.elements[tab_idx]
    
    def update(self):
        self.current_delta = self.clock.get_time() / 1000.0
        
        if self.mouse_down:
            mouse_pos = pygame.mouse.get_pos()
            # Adjust mouse position relative to simulation area
            adjusted_pos = (
                mouse_pos[0] - self.sidebar_width,
                mouse_pos[1] - self.tab_height
            )
            self.particle_system.create_particle_burst(
                adjusted_pos,
                self.current_delta,
                self.selected_element,
                self.particles_per_burst
            )
        
        self.particle_system.update(self.current_delta)
    
    def _draw_sidebar(self):
        """Draw the sidebar with stats and info"""
        self.sidebar.fill(UI_COLORS['PANEL'])
        
        # Draw stats
        y_offset = self.ui_padding * 2
        stats = [
            f'FPS: {int(self.clock.get_fps())}',
            f'Particles: {self.particle_system.active_particles}',
            f'Selected: {self.selected_element}',
            f'Burst Size: {self.particles_per_burst}/click'  # Add burst size display
        ]
        
        for stat in stats:
            text = self.stats_font.render(stat, True, UI_COLORS['TEXT'])
            # Add subtle text shadow
            shadow = self.stats_font.render(stat, True, (0, 0, 0))
            self.sidebar.blit(shadow, (self.ui_padding + 1, y_offset + 1))
            self.sidebar.blit(text, (self.ui_padding, y_offset))
            y_offset += 35
        
        # Draw a subtle gradient border
        for i in range(2):
            color = (UI_COLORS['BORDER'][0], 
                    UI_COLORS['BORDER'][1],
                    UI_COLORS['BORDER'][2],
                    150 - i * 50)
            pygame.draw.line(self.sidebar, color,
                            (self.sidebar_width - i, 0),
                            (self.sidebar_width - i, WINDOW_HEIGHT))
    
    def _draw_element_tabs(self):
        """Draw the element selection tabs"""
        self.element_tabs.fill(UI_COLORS['PANEL'])
        
        mouse_pos = pygame.mouse.get_pos()
        adjusted_mouse_x = mouse_pos[0] - self.sidebar_width
        
        for i, element in enumerate(self.elements):
            x = i * (self.tab_width + self.tab_padding)
            tab_rect = pygame.Rect(x, 2, self.tab_width, self.tab_height - 4)
            
            # Determine tab state and colors
            is_selected = element == self.selected_element
            is_hovered = (tab_rect.collidepoint(adjusted_mouse_x, mouse_pos[1]) and 
                         mouse_pos[1] < self.tab_height)
            
            if is_selected:
                bg_color = ELEMENT_COLORS[element]
                text_color = UI_COLORS['PANEL']
            elif is_hovered:
                bg_color = UI_COLORS['TAB_HOVER']
                text_color = ELEMENT_COLORS[element]
            else:
                bg_color = UI_COLORS['PANEL']
                text_color = ELEMENT_COLORS[element]
            
            # Draw rounded tab background
            pygame.draw.rect(self.element_tabs, bg_color, tab_rect, 
                            border_radius=self.corner_radius)
            
            if not is_selected:
                # Draw subtle border for unselected tabs
                pygame.draw.rect(self.element_tabs, UI_COLORS['BORDER'], tab_rect, 
                               1, border_radius=self.corner_radius)
            
            # Draw element symbol
            text = self.element_font.render(element, True, text_color)
            text_rect = text.get_rect(center=(x + self.tab_width/2, self.tab_height/2))
            
            # Add subtle glow effect for selected tab
            if is_selected:
                glow = self.element_font.render(element, True, (255, 255, 255, 128))
                glow_rect = glow.get_rect(center=text_rect.center)
                self.element_tabs.blit(glow, (glow_rect.x + 1, glow_rect.y + 1))
            
            self.element_tabs.blit(text, text_rect)
        
        # Draw bottom border with gradient
        for i in range(2):
            color = (UI_COLORS['BORDER'][0],
                    UI_COLORS['BORDER'][1],
                    UI_COLORS['BORDER'][2],
                    150 - i * 50)
            pygame.draw.line(self.element_tabs, color,
                            (0, self.tab_height - i - 1),
                            (WINDOW_WIDTH - self.sidebar_width, self.tab_height - i - 1))
    
    def draw(self):
        self.screen.fill(UI_COLORS['BACKGROUND'])
        
        # Draw sidebar
        self._draw_sidebar()
        self.screen.blit(self.sidebar, (0, 0))
        
        # Draw element tabs
        self._draw_element_tabs()
        self.screen.blit(self.element_tabs, (self.sidebar_width, 0))
        
        # Draw simulation area with subtle border
        self.simulation_area.fill(UI_COLORS['BACKGROUND'])
        self.particle_system.draw(self.simulation_area)
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