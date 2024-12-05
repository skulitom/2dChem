import sys
import pygame
from pygame.locals import *
from pygame import Surface

from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    BLACK, WHITE, ELEMENT_COLORS,
    UI_COLORS, INTERACTION_MODES,
    SIMULATION_X_OFFSET, SIMULATION_Y_OFFSET
)
from physics.particle_system import ParticleSystem
from utils.profiler import Profiler
from ui.ui_manager import UIManager

class Simulation:
    def __init__(self):
        pygame.init()
        
        # Initialize display first
        self._init_display()
        
        # Then UI configuration
        self._init_ui_config()
        
        # Initialize remaining components
        self._init_state()
        
        # Initialize surfaces after we know sidebar and tab dimensions
        self._init_surfaces()
        
        self.profiler = Profiler()
        self.profiler.start()
        
        self.interaction_mode = INTERACTION_MODES['CREATE']  # Default to create mode
    
    def _init_display(self):
        """Initialize display and timing components"""
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2D Chemistry Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
    
    def _init_ui_config(self):
        """Initialize UI configuration values"""
        self.ui_manager = UIManager(self)
        self.sidebar_width = SIMULATION_X_OFFSET
        self.tab_height = SIMULATION_Y_OFFSET
        
        # Define UI metrics and fonts
        self.ui_padding = 10
        self.tab_width = 50
        self.tab_padding = 10
        self.corner_radius = 5
        self.button_height = 40
        self.elements = ['H', 'O', 'C', 'N']  # Example elements
        self.stats_font = pygame.font.Font(None, 24)
        self.element_font = pygame.font.Font(None, 24)
    
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
            
            # Handle scroll events before UI manager
            if event.type == MOUSEBUTTONDOWN and event.button in (4, 5):
                if event.button == 4:  # Mouse wheel up
                    self.particles_per_burst = min(100, self.particles_per_burst + 5)
                else:  # Mouse wheel down
                    self.particles_per_burst = max(1, self.particles_per_burst - 5)
                continue
            
            # Let UI handle other events
            if self.ui_manager.handle_event(event):
                continue
            
            # Handle remaining simulation events
            if event.type == MOUSEBUTTONDOWN:
                return self._handle_mouse_down(event)
            elif event.type == MOUSEBUTTONUP:
                self._handle_mouse_up(event)
            elif event.type == MOUSEMOTION:
                self._handle_mouse_motion(event)
        return True
    
    def _handle_mouse_down(self, event):
        """Handle mouse down event"""
        mouse_pos = pygame.mouse.get_pos()
        
        # Let UI manager handle the click first
        if self.ui_manager.handle_event(event):
            return True
        
        # Handle simulation area clicks
        if self._is_in_simulation_area(mouse_pos):
            if self.interaction_mode == INTERACTION_MODES['CREATE']:
                self.mouse_down = True
            else:  # DRAG mode
                # Adjust mouse position to simulation coordinates
                adjusted_pos = (
                    mouse_pos[0] - SIMULATION_X_OFFSET,
                    mouse_pos[1] - SIMULATION_Y_OFFSET
                )
                self.particle_system.physics.start_drag(self.particle_system, adjusted_pos)
        
        return False
    
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
        if 0 <= tab_idx < len(self.elements):
            self.selected_element = self.elements[tab_idx]
    
    def _handle_mouse_up(self, event):
        """Handle mouse up event"""
        self.mouse_down = False
        self.particle_system.physics.end_drag(self.particle_system)
    
    def _handle_mouse_motion(self, event):
        """Handle mouse motion event"""
        if self.interaction_mode == INTERACTION_MODES['DRAG']:
            # Adjust mouse position to simulation coordinates
            adjusted_pos = (
                event.pos[0] - SIMULATION_X_OFFSET,
                event.pos[1] - SIMULATION_Y_OFFSET
            )
            self.particle_system.physics.update_drag(self.particle_system, adjusted_pos)
    
    def update(self):
        """Update simulation state"""
        self.current_delta = self.clock.get_time() / 1000.0
        
        # Only create particles while mouse is held down
        if self.mouse_down and self.interaction_mode == INTERACTION_MODES['CREATE']:
            mouse_pos = pygame.mouse.get_pos()
            # Check if inside simulation area via UI manager
            if hasattr(self.ui_manager, 'simulation_area') and self.ui_manager.simulation_area.is_point_inside(mouse_pos):
                # Adjust mouse position relative to simulation area
                adjusted_pos = (
                    mouse_pos[0] - SIMULATION_X_OFFSET,
                    mouse_pos[1] - SIMULATION_Y_OFFSET
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
        self.sidebar.fill((0, 0, 0, 0))
        
        # Draw main panel with gradient
        gradient_rect = pygame.Surface((self.sidebar_width, WINDOW_HEIGHT), pygame.SRCALPHA)
        for i in range(WINDOW_HEIGHT):
            alpha = min(240, 180 + i // 4)  # Subtle vertical gradient
            color = (*UI_COLORS['PANEL'], alpha)
            pygame.draw.line(gradient_rect, color, (0, i), (self.sidebar_width, i))
        self.sidebar.blit(gradient_rect, (0, 0))
        
        # Draw stats
        y_offset = self.ui_padding * 2
        stats = [
            f'FPS: {int(self.clock.get_fps())}',
            f'Particles: {self.particle_system.active_particles}',
            f'Selected: {self.selected_element}',
            f'Burst Size: {self.particles_per_burst}/click'
        ]
        
        for stat in stats:
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
            
            text = self.stats_font.render(stat, True, UI_COLORS['TEXT'])
            self.sidebar.blit(text, (self.ui_padding, y_offset))
            
            y_offset += 40

        # Clear button
        self.clear_button_rect = pygame.Rect(
            self.ui_padding,
            y_offset + 10,
            self.sidebar_width - (self.ui_padding * 2),
            self.button_height
        )
        
        gradient_surface = pygame.Surface(self.clear_button_rect.size, pygame.SRCALPHA)
        height = self.clear_button_rect.height
        for i in range(height):
            progress = i / height
            alpha = int(255 * (1 - progress * 0.2))  # Subtler gradient
            color = (UI_COLORS['TAB_HOVER'] if self.clear_button_rect.collidepoint(pygame.mouse.get_pos()) 
                     else UI_COLORS['PANEL'])
            color = (*color, alpha)
            pygame.draw.line(gradient_surface, color, 
                             (0, i), (self.clear_button_rect.width, i))
        
        self.sidebar.blit(gradient_surface, self.clear_button_rect)
        pygame.draw.rect(self.sidebar, UI_COLORS['BORDER'], self.clear_button_rect, 1,
                         border_radius=self.corner_radius)
        
        clear_text = self.stats_font.render("Clear Particles", True, 
                                            UI_COLORS['HIGHLIGHT'] if self.clear_button_rect.collidepoint(pygame.mouse.get_pos()) 
                                            else UI_COLORS['TEXT'])
        text_rect = clear_text.get_rect(center=self.clear_button_rect.center)
        self.sidebar.blit(clear_text, text_rect)
        
        # Mode switch button (define rect before use)
        self.mode_button_rect = pygame.Rect(
            self.ui_padding,
            self.clear_button_rect.bottom + 20,
            self.sidebar_width - (self.ui_padding * 2),
            self.button_height
        )
        
        pygame.draw.rect(self.sidebar, UI_COLORS['PANEL'], self.mode_button_rect)
        mode_text = f"Mode: {'Drag' if self.interaction_mode == INTERACTION_MODES['DRAG'] else 'Create'}"
        mode_surface = self.font.render(mode_text, True, UI_COLORS['TEXT'])
        mode_rect = mode_surface.get_rect(center=self.mode_button_rect.center)
        self.sidebar.blit(mode_surface, mode_rect)
    
    def _draw_element_tabs(self):
        """Draw the element selection tabs"""
        self.element_tabs.fill((0,0,0,0))
        
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
            
            self.element_tabs.blit(gradient_surface, tab_rect)
            
            # Draw tab border for non-selected tabs
            if not is_selected:
                pygame.draw.rect(self.element_tabs, UI_COLORS['BORDER'],
                                 tab_rect, 1, border_radius=self.corner_radius)
            
            # Text color
            if is_selected:
                text_color = UI_COLORS['TEXT']
            elif is_hovered:
                text_color = (*UI_COLORS['TEXT'], 220)
            else:
                text_color = ELEMENT_COLORS[element]
                
            text = self.element_font.render(element, True, text_color)
            text_rect = text.get_rect(center=(x + self.tab_width/2, self.tab_height/2))
            self.element_tabs.blit(text, text_rect)
    
    def draw(self):
        """Draw the simulation"""
        self.screen.fill(BLACK)
        
        # Draw the sidebar and element tabs
        self._draw_sidebar()
        self._draw_element_tabs()
        
        # Draw particles on simulation_area surface
        self.simulation_area.fill(BLACK)
        self.particle_system.draw(self.simulation_area)
        
        # Blit surfaces to the screen
        self.screen.blit(self.sidebar, (0, 0))
        self.screen.blit(self.element_tabs, (self.sidebar_width, 0))
        self.screen.blit(self.simulation_area, (self.sidebar_width, self.tab_height))
        
        # Finally, draw the UI manager components (if any)
        self.ui_manager.draw()
        
        # Update the display
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
    
    def toggle_interaction_mode(self):
        """Toggle between CREATE and DRAG interaction modes"""
        self.interaction_mode = (
            INTERACTION_MODES['DRAG'] 
            if self.interaction_mode == INTERACTION_MODES['CREATE'] 
            else INTERACTION_MODES['CREATE']
        )

if __name__ == '__main__':
    simulation = Simulation()
    simulation.run()
