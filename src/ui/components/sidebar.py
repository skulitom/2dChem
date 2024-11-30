import pygame
from core.constants import UI_COLORS, SIMULATION_X_OFFSET, WINDOW_HEIGHT, INTERACTION_MODES
from .base import UIComponent

class Sidebar(UIComponent):
    def __init__(self, simulation):
        rect = pygame.Rect(0, 0, SIMULATION_X_OFFSET, WINDOW_HEIGHT)
        super().__init__(simulation, rect)
        
        self.ui_padding = 12
        self.button_height = 35
        self.stats_font = pygame.font.Font(None, 26)
        
        # Calculate button positions
        stats_height = self.ui_padding * 2 + (40 * 4)  # 4 stats * 40 pixels each
        
        # Store button rectangles
        self.clear_button_rect = pygame.Rect(
            self.ui_padding,
            stats_height + 10,
            self.rect.width - (self.ui_padding * 2),
            self.button_height
        )
        
        self.mode_button_rect = pygame.Rect(
            self.ui_padding,
            self.clear_button_rect.bottom + 10,
            self.rect.width - (self.ui_padding * 2),
            self.button_height
        )
        
    def draw(self, screen):
        # Clear surface
        self.surface.fill(UI_COLORS['PANEL'])
        
        # Draw stats
        y_offset = self.ui_padding * 2
        stats = [
            f'FPS: {int(self.simulation.clock.get_fps())}',
            f'Particles: {self.simulation.particle_system.active_particles}',
            f'Selected: {self.simulation.selected_element}',
            f'Burst Size: {self.simulation.particles_per_burst}/click'
        ]
        
        for stat in stats:
            text = self.stats_font.render(stat, True, UI_COLORS['TEXT'])
            self.surface.blit(text, (self.ui_padding, y_offset))
            y_offset += 40
            
        # Draw buttons using stored rectangles
        pygame.draw.rect(self.surface, UI_COLORS['TAB_ACTIVE'], self.clear_button_rect)
        text = self.stats_font.render("Clear Particles", True, UI_COLORS['TEXT'])
        text_rect = text.get_rect(center=self.clear_button_rect.center)
        self.surface.blit(text, text_rect)
        
        pygame.draw.rect(self.surface, UI_COLORS['TAB_ACTIVE'], self.mode_button_rect)
        mode_text = f"Mode: {'Drag' if self.simulation.interaction_mode == INTERACTION_MODES['DRAG'] else 'Create'}"
        text = self.stats_font.render(mode_text, True, UI_COLORS['TEXT'])
        text_rect = text.get_rect(center=self.mode_button_rect.center)
        self.surface.blit(text, text_rect)
        
        # Draw to screen
        screen.blit(self.surface, self.rect)
        
    def handle_click(self, pos):
        if not self.is_point_inside(pos):
            return False
            
        # Convert to local coordinates
        local_pos = (pos[0] - self.rect.x, pos[1] - self.rect.y)
        
        # Use stored button rectangles for hit testing
        if self.clear_button_rect.collidepoint(local_pos):
            self.simulation.particle_system.clear_particles()
            return True
            
        if self.mode_button_rect.collidepoint(local_pos):
            # Toggle mode and ensure mouse state is reset
            self.simulation.toggle_interaction_mode()
            self.simulation.mouse_down = False
            return True
            
        return False 