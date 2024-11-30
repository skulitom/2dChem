import sys
import pygame
from pygame.locals import *
from pygame import Surface

from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    BLACK, WHITE, ELEMENT_COLORS,
    UI_COLORS
)
from physics.particle_system import ParticleSystem
from utils.profiler import Profiler

class Simulation:
    def __init__(self):
        pygame.init()
        
        # UI configuration
        self._init_ui_config()
        
        # Initialize core components
        self._init_display()
        self._init_state()
        self._init_surfaces()
        self.profiler = Profiler()
        self.profiler.start()
    
    def _init_ui_config(self):
        """Initialize UI configuration values"""
        self.ui_padding = 12
        self.tab_height = 45
        self.tab_width = 75
        self.tab_padding = 3
        self.sidebar_width = 220
        self.elements = ['H', 'O', 'N', 'C']
        self.element_font = pygame.font.Font(None, 32)
        self.stats_font = pygame.font.Font(None, 26)
        self.corner_radius = 8
        self.button_height = 35
    
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
        self.particles_per_burst = 10  # Default particles per burst
    
    def _init_surfaces(self):
        """Initialize UI surfaces"""
        self.sidebar = Surface((self.sidebar_width, WINDOW_HEIGHT))
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
        
        # Check if clear button was clicked
        button_rect = pygame.Rect(
            self.ui_padding,
            self.ui_padding * 2 + 35 * 4 + 10,  # Position after stats
            self.sidebar_width - (self.ui_padding * 2),
            self.button_height
        )
        
        if button_rect.collidepoint(mouse_pos):
            self.particle_system.clear_particles()
            return
        
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
                pos=adjusted_pos,
                element_type=self.selected_element,
                burst_size=self.particles_per_burst,
                spread=20.0,
                speed=2.0
            )
        
        self.particle_system.update(self.current_delta)
    
    def _draw_sidebar(self):
        """Draw the sidebar with stats and info"""
        # Draw main panel with gradient
        gradient_rect = pygame.Surface((self.sidebar_width, WINDOW_HEIGHT), pygame.SRCALPHA)
        for i in range(WINDOW_HEIGHT):
            alpha = min(240, 180 + i // 4)  # Subtle vertical gradient
            color = (*UI_COLORS['PANEL'], alpha)
            pygame.draw.line(gradient_rect, color, (0, i), (self.sidebar_width, i))
        self.sidebar.blit(gradient_rect, (0, 0))
        
        # Draw stats with enhanced styling
        y_offset = self.ui_padding * 2
        stats = [
            f'FPS: {int(self.clock.get_fps())}',
            f'Particles: {self.particle_system.active_particles}',
            f'Selected: {self.selected_element}',
            f'Burst Size: {self.particles_per_burst}/click'
        ]
        
        for stat in stats:
            # Draw stat background with subtle gradient
            stat_rect = pygame.Rect(
                self.ui_padding - 4,
                y_offset - 4,
                self.sidebar_width - (self.ui_padding * 2) + 8,
                30
            )
            
            # Gradient background for stats
            gradient_surface = pygame.Surface(stat_rect.size, pygame.SRCALPHA)
            for i in range(stat_rect.height):
                alpha = 150 - (i // 2)
                pygame.draw.line(gradient_surface, (*UI_COLORS['PANEL'], alpha),
                               (0, i), (stat_rect.width, i))
            self.sidebar.blit(gradient_surface, stat_rect)
            
            # Clean text rendering without glow
            text = self.stats_font.render(stat, True, UI_COLORS['TEXT'])
            self.sidebar.blit(text, (self.ui_padding, y_offset))
            
            y_offset += 40

        # Draw Clear button with refined styling
        button_rect = pygame.Rect(
            self.ui_padding,
            y_offset + 10,
            self.sidebar_width - (self.ui_padding * 2),
            self.button_height
        )
        
        mouse_pos = pygame.mouse.get_pos()
        button_hovered = button_rect.collidepoint(mouse_pos)
        
        # Smoother button gradient
        gradient_surface = pygame.Surface(button_rect.size, pygame.SRCALPHA)
        height = button_rect.height
        for i in range(height):
            progress = i / height
            alpha = int(255 * (1 - progress * 0.2))  # Subtler gradient
            color = UI_COLORS['TAB_HOVER'] if button_hovered else UI_COLORS['PANEL']
            color = (*color, alpha)
            pygame.draw.line(gradient_surface, color, 
                            (0, i), (button_rect.width, i))
        
        # Apply gradient and border
        self.sidebar.blit(gradient_surface, button_rect)
        pygame.draw.rect(self.sidebar, UI_COLORS['BORDER'], button_rect, 1,
                        border_radius=self.corner_radius)
        
        # Clean button text rendering
        clear_text = self.stats_font.render("Clear Particles", True, 
                                          UI_COLORS['HIGHLIGHT'] if button_hovered else UI_COLORS['TEXT'])
        text_rect = clear_text.get_rect(center=button_rect.center)
        self.sidebar.blit(clear_text, text_rect)
    
    def _draw_element_tabs(self):
        """Draw the element selection tabs with enhanced styling"""
        # Draw tab panel background with gradient
        gradient_rect = pygame.Surface((WINDOW_WIDTH - self.sidebar_width, self.tab_height), pygame.SRCALPHA)
        for i in range(self.tab_height):
            alpha = min(240, 180 + i // 2)
            color = (*UI_COLORS['PANEL'], alpha)
            pygame.draw.line(gradient_rect, color,
                            (0, i), (WINDOW_WIDTH - self.sidebar_width, i))
        self.element_tabs.blit(gradient_rect, (0, 0))
        
        mouse_pos = pygame.mouse.get_pos()
        adjusted_mouse_x = mouse_pos[0] - self.sidebar_width
        
        for i, element in enumerate(self.elements):
            x = i * (self.tab_width + self.tab_padding)
            tab_rect = pygame.Rect(x, 2, self.tab_width, self.tab_height - 4)
            
            # Determine tab state
            is_selected = element == self.selected_element
            is_hovered = (tab_rect.collidepoint(adjusted_mouse_x, mouse_pos[1]) and 
                         mouse_pos[1] < self.tab_height)
            
            # Refined tab gradient
            gradient_surface = pygame.Surface(tab_rect.size, pygame.SRCALPHA)
            height = tab_rect.height
            for y in range(height):
                progress = y / height
                alpha = int(255 * (1 - progress * 0.3))  # Subtler gradient
                
                if is_selected:
                    color = (*ELEMENT_COLORS[element], alpha)
                elif is_hovered:
                    color = (*UI_COLORS['TAB_HOVER'], alpha)
                else:
                    color = (*UI_COLORS['PANEL'], alpha)
                    
                pygame.draw.line(gradient_surface, color,
                               (0, y), (tab_rect.width, y))
            
            # Apply gradient
            self.element_tabs.blit(gradient_surface, tab_rect)
            
            # Draw tab border for non-selected tabs
            if not is_selected:
                pygame.draw.rect(self.element_tabs, UI_COLORS['BORDER'],
                               tab_rect, 1, border_radius=self.corner_radius)
            
            # Clean text rendering for element symbols
            if is_selected:
                text_color = UI_COLORS['TEXT']
            elif is_hovered:
                text_color = (*UI_COLORS['TEXT'], 220)  # Slightly dimmed on hover
            else:
                text_color = ELEMENT_COLORS[element]
                
            text = self.element_font.render(element, True, text_color)
            text_rect = text.get_rect(center=(x + self.tab_width/2, self.tab_height/2))
            self.element_tabs.blit(text, text_rect)
    
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
        
        self.profiler.stop()  # Stop profiling before exit
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    simulation = Simulation()
    simulation.run() 