import pygame
from core.constants import UI_COLORS, SIMULATION_X_OFFSET, WINDOW_HEIGHT, INTERACTION_MODES
from .base import UIComponent

class Sidebar(UIComponent):
    """
    A sidebar UI component that displays simulation stats and provides
    buttons to clear particles and toggle interaction modes.
    """
    def __init__(self, simulation):
        rect = pygame.Rect(0, 0, SIMULATION_X_OFFSET, WINDOW_HEIGHT)
        super().__init__(simulation, rect)
        
        # Padding and sizing
        self.ui_padding = 12
        self.button_height = 35
        self.stats_font = pygame.font.Font(None, 26)
        
        # Determine vertical space taken by stats
        self.num_stats = 4
        stats_height = self.ui_padding * 2 + (40 * self.num_stats)
        
        # Button rectangles
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
        """
        Draws the sidebar including stats and buttons onto the screen.
        """
        # Draw sidebar background with a subtle vertical gradient
        self._draw_gradient_background()
        
        # Draw simulation stats
        self._draw_stats()
        
        # Draw interactive buttons with hover states
        self._draw_button(
            self.clear_button_rect,
            "Clear Particles",
            hover_color=UI_COLORS['TAB_HOVER'],
            default_color=UI_COLORS['TAB_ACTIVE']
        )
        
        mode_text = f"Mode: {'Drag' if self.simulation.interaction_mode == INTERACTION_MODES['DRAG'] else 'Create'}"
        self._draw_button(
            self.mode_button_rect,
            mode_text,
            hover_color=UI_COLORS['TAB_HOVER'],
            default_color=UI_COLORS['TAB_ACTIVE']
        )
        
        # Finally, blit sidebar surface to the main screen
        screen.blit(self.surface, self.rect)
        
    def _draw_gradient_background(self):
        """
        Draw a subtle vertical gradient for the sidebar background.
        """
        self.surface.fill((0,0,0,0))
        gradient_rect = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        for i in range(self.rect.height):
            alpha = min(240, 180 + i // 4)  # Subtle vertical gradient
            color = (*UI_COLORS['PANEL'], alpha)
            pygame.draw.line(gradient_rect, color, (0, i), (self.rect.width, i))
        self.surface.blit(gradient_rect, (0, 0))
    
    def _draw_stats(self):
        """
        Draw simulation statistics on the sidebar.
        """
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
    
    def _draw_button(self, rect, label, hover_color, default_color):
        """
        Draw a button with a subtle gradient and hover state.

        :param rect: The button rectangle (in local coordinates).
        :param label: The text to display on the button.
        :param hover_color: The color used if hovered.
        :param default_color: The default color of the button.
        """
        mouse_pos = pygame.mouse.get_pos()
        local_mouse_pos = (mouse_pos[0] - self.rect.x, mouse_pos[1] - self.rect.y)
        
        # Check hover state
        is_hovered = rect.collidepoint(local_mouse_pos)
        
        # Choose button color based on hover
        base_color = hover_color if is_hovered else default_color
        # Create a gradient surface for the button
        button_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        for i in range(rect.height):
            progress = i / rect.height
            alpha = int(255 * (1 - progress * 0.2))
            color = (*base_color, alpha)
            pygame.draw.line(button_surface, color, (0, i), (rect.width, i))
        
        # Draw button border
        pygame.draw.rect(button_surface, UI_COLORS['BORDER'], button_surface.get_rect(), 1)
        self.surface.blit(button_surface, rect)
        
        # Draw button label
        text = self.stats_font.render(label, True, UI_COLORS['TEXT'])
        text_rect = text.get_rect(center=rect.center)
        self.surface.blit(text, text_rect)
    
    def handle_click(self, pos):
        """
        Handles clicks within the sidebar area.

        :param pos: The global mouse position.
        :return: True if the click was handled, False otherwise.
        """
        if not self.is_point_inside(pos):
            return False
            
        # Convert to local coordinates
        local_pos = (pos[0] - self.rect.x, pos[1] - self.rect.y)
        
        # Check if clear button was clicked
        if self.clear_button_rect.collidepoint(local_pos):
            self.simulation.particle_system.clear_particles()
            return True
            
        # Check if mode button was clicked
        if self.mode_button_rect.collidepoint(local_pos):
            # Toggle mode and ensure mouse state is reset
            self.simulation.toggle_interaction_mode()
            self.simulation.mouse_down = False
            return True
            
        return False
