import pygame
from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    UI_COLORS, SIMULATION_X_OFFSET, SIMULATION_Y_OFFSET,
    INTERACTION_MODES
)
from .components import Sidebar, ElementTabs, SimulationArea

class UIManager:
    def __init__(self, simulation):
        self.simulation = simulation
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2D Chemistry Simulation")
        
        # Initialize components
        self.sidebar = Sidebar(simulation)
        self.element_tabs = ElementTabs(simulation)
        self.simulation_area = SimulationArea(simulation)
        
    def handle_event(self, event):
        """Handle UI events and return True if event was handled."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            # Check components in order of precedence
            if self.sidebar.handle_click(mouse_pos):
                return True
            if self.element_tabs.handle_click(mouse_pos):
                return True
            if self.simulation_area.handle_click(mouse_pos):
                return True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            # Reset mouse state
            self.simulation.mouse_down = False
            if self.simulation.interaction_mode == INTERACTION_MODES['DRAG']:
                self.simulation.particle_system.physics.end_drag(self.simulation.particle_system)
                
        elif event.type == pygame.MOUSEMOTION:
            if self.simulation.mouse_down:
                mouse_pos = pygame.mouse.get_pos()
                if self.simulation.interaction_mode == INTERACTION_MODES['DRAG']:
                    # Convert to simulation coordinates for drag
                    if mouse_pos[0] >= SIMULATION_X_OFFSET:
                        sim_pos = (
                            mouse_pos[0] - SIMULATION_X_OFFSET,
                            mouse_pos[1] - SIMULATION_Y_OFFSET
                        )
                        self.simulation.particle_system.physics.update_drag(
                            self.simulation.particle_system, 
                            sim_pos
                        )
                    
        return False
        
    def draw(self):
        """Draw all UI components"""
        # Clear screen
        self.screen.fill(UI_COLORS['BACKGROUND'])
        
        # Draw components
        self.sidebar.draw(self.screen)
        self.element_tabs.draw(self.screen)
        self.simulation_area.draw(self.screen)
        
        # Update display
        pygame.display.flip() 