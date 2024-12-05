import pygame
from core.constants import UI_COLORS, ELEMENT_COLORS, SIMULATION_X_OFFSET, WINDOW_WIDTH
from .base import UIComponent

class ElementTabs(UIComponent):
    """
    A UI component that displays element tabs at the top of the simulation area.
    Allows users to select different element types.
    """
    def __init__(self, simulation):
        # Set a fixed height for the tab strip
        tab_height = 45
        rect = pygame.Rect(
            SIMULATION_X_OFFSET, 
            0, 
            WINDOW_WIDTH - SIMULATION_X_OFFSET,
            tab_height
        )
        super().__init__(simulation, rect)
        
        # Tab styling
        self.tab_width = 75
        self.tab_padding = 3
        self.corner_radius = 8
        self.element_font = pygame.font.Font(None, 32)
        
        # Define the available elements
        self.elements = ['H', 'O', 'N', 'C']
    
    def draw(self, screen):
        """
        Draw the element tabs onto the screen surface.
        """
        # Draw background gradient
        self._draw_gradient_background()
        
        # Get mouse position and adjust relative to element tab area
        mouse_pos = pygame.mouse.get_pos()
        adjusted_mouse_x = mouse_pos[0] - self.rect.x
        
        # Draw each element tab
        for i, element in enumerate(self.elements):
            x = i * (self.tab_width + self.tab_padding)
            tab_rect = pygame.Rect(x, 2, self.tab_width, self.rect.height - 4)
            
            is_selected = (element == self.simulation.selected_element)
            is_hovered = (tab_rect.collidepoint(adjusted_mouse_x, mouse_pos[1]) and 
                          self.rect.collidepoint(mouse_pos))
            
            self._draw_tab(element, tab_rect, is_selected, is_hovered)
        
        # Blit to the main screen
        screen.blit(self.surface, self.rect)
    
    def _draw_gradient_background(self):
        """
        Draw a subtle vertical gradient in the background of the tab area.
        """
        self.surface.fill((0,0,0,0))
        gradient_rect = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        for i in range(self.rect.height):
            alpha = min(240, 180 + i // 2)
            color = (*UI_COLORS['PANEL'], alpha)
            pygame.draw.line(gradient_rect, color, (0, i), (self.rect.width, i))
        self.surface.blit(gradient_rect, (0, 0))
    
    def _draw_tab(self, element, tab_rect, is_selected, is_hovered):
        """
        Draw a single tab representing an element.
        
        :param element: The chemical symbol for the element (e.g., 'H', 'O', 'C', 'N').
        :param tab_rect: The rectangle defining the tab's position and size.
        :param is_selected: Boolean indicating if this tab is the currently selected element.
        :param is_hovered: Boolean indicating if the mouse is currently hovering over this tab.
        """
        # Create a gradient for the tab background
        gradient_surface = pygame.Surface(tab_rect.size, pygame.SRCALPHA)
        height = tab_rect.height
        
        for y in range(height):
            progress = y / height
            alpha = int(255 * (1 - progress * 0.3))
            
            if is_selected:
                # Selected tabs use their element color
                color = (*ELEMENT_COLORS[element], alpha)
            elif is_hovered:
                # Hovered tabs use a hover color
                color = (*UI_COLORS['TAB_HOVER'], alpha)
            else:
                # Default state
                color = (*UI_COLORS['PANEL'], alpha)
                
            pygame.draw.line(gradient_surface, color, (0, y), (tab_rect.width, y))
        
        # Blit the gradient surface for the tab
        self.surface.blit(gradient_surface, tab_rect)
        
        # Draw a border around non-selected tabs
        if not is_selected:
            pygame.draw.rect(self.surface, UI_COLORS['BORDER'],
                             tab_rect, 1, border_radius=self.corner_radius)
        
        # Determine text color
        if is_selected:
            text_color = UI_COLORS['TEXT']
        elif is_hovered:
            # Slightly dim text on hover if not selected
            text_color = (*UI_COLORS['TEXT'], 220)
        else:
            # Use element color for unselected, unhovered state
            text_color = ELEMENT_COLORS[element]
        
        # Render and draw the element symbol text centered in the tab
        text = self.element_font.render(element, True, text_color)
        text_rect = text.get_rect(center=tab_rect.center)
        self.surface.blit(text, text_rect)
    
    def handle_click(self, pos):
        """
        Handle mouse clicks on the element tabs.
        
        :param pos: Global mouse position.
        :return: True if an element tab was clicked and handled, False otherwise.
        """
        if not self.is_point_inside(pos):
            return False
            
        # Convert to local coordinates
        local_x = pos[0] - self.rect.x
        
        # Determine which tab was clicked
        tab_idx = local_x // (self.tab_width + self.tab_padding)
        if 0 <= tab_idx < len(self.elements):
            self.simulation.selected_element = self.elements[tab_idx]
            return True
            
        return False
