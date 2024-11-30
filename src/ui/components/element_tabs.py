import pygame
from core.constants import UI_COLORS, ELEMENT_COLORS, SIMULATION_X_OFFSET, WINDOW_WIDTH
from .base import UIComponent

class ElementTabs(UIComponent):
    def __init__(self, simulation):
        rect = pygame.Rect(
            SIMULATION_X_OFFSET, 
            0, 
            WINDOW_WIDTH - SIMULATION_X_OFFSET,
            45  # tab height
        )
        super().__init__(simulation, rect)
        
        self.tab_width = 75
        self.tab_padding = 3
        self.corner_radius = 8
        self.element_font = pygame.font.Font(None, 32)
        self.elements = ['H', 'O', 'N', 'C']
        
    def draw(self, screen):
        # Draw tab panel background with gradient
        gradient_rect = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        for i in range(self.rect.height):
            alpha = min(240, 180 + i // 2)
            color = (*UI_COLORS['PANEL'], alpha)
            pygame.draw.line(gradient_rect, color, (0, i), (self.rect.width, i))
        self.surface.blit(gradient_rect, (0, 0))
        
        mouse_pos = pygame.mouse.get_pos()
        adjusted_mouse_x = mouse_pos[0] - self.rect.x
        
        for i, element in enumerate(self.elements):
            x = i * (self.tab_width + self.tab_padding)
            tab_rect = pygame.Rect(x, 2, self.tab_width, self.rect.height - 4)
            
            # Determine tab state
            is_selected = element == self.simulation.selected_element
            is_hovered = (tab_rect.collidepoint(adjusted_mouse_x, mouse_pos[1]) and 
                         self.rect.collidepoint(mouse_pos))
            
            # Create tab gradient
            gradient_surface = pygame.Surface(tab_rect.size, pygame.SRCALPHA)
            height = tab_rect.height
            for y in range(height):
                progress = y / height
                alpha = int(255 * (1 - progress * 0.3))
                
                if is_selected:
                    color = (*ELEMENT_COLORS[element], alpha)
                elif is_hovered:
                    color = (*UI_COLORS['TAB_HOVER'], alpha)
                else:
                    color = (*UI_COLORS['PANEL'], alpha)
                    
                pygame.draw.line(gradient_surface, color, (0, y), (tab_rect.width, y))
            
            # Apply gradient
            self.surface.blit(gradient_surface, tab_rect)
            
            # Draw tab border for non-selected tabs
            if not is_selected:
                pygame.draw.rect(self.surface, UI_COLORS['BORDER'],
                               tab_rect, 1, border_radius=self.corner_radius)
            
            # Render element text
            if is_selected:
                text_color = UI_COLORS['TEXT']
            elif is_hovered:
                text_color = (*UI_COLORS['TEXT'], 220)
            else:
                text_color = ELEMENT_COLORS[element]
                
            text = self.element_font.render(element, True, text_color)
            text_rect = text.get_rect(center=(x + self.tab_width/2, self.rect.height/2))
            self.surface.blit(text, text_rect)
            
        screen.blit(self.surface, self.rect)
        
    def handle_click(self, pos):
        if not self.is_point_inside(pos):
            return False
            
        # Convert to local coordinates
        local_x = pos[0] - self.rect.x
        
        # Calculate which tab was clicked
        tab_idx = local_x // (self.tab_width + self.tab_padding)
        if tab_idx < len(self.elements):
            self.simulation.selected_element = self.elements[tab_idx]
            return True
            
        return False 