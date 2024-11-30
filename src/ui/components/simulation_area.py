import pygame
from core.constants import (
    UI_COLORS, SIMULATION_X_OFFSET, SIMULATION_Y_OFFSET,
    WINDOW_WIDTH, WINDOW_HEIGHT, INTERACTION_MODES
)
from .base import UIComponent

class SimulationArea(UIComponent):
    def __init__(self, simulation):
        rect = pygame.Rect(
            SIMULATION_X_OFFSET,
            SIMULATION_Y_OFFSET,
            WINDOW_WIDTH - SIMULATION_X_OFFSET,
            WINDOW_HEIGHT - SIMULATION_Y_OFFSET
        )
        super().__init__(simulation, rect)
        
    def draw(self, screen):
        # Clear simulation area
        self.surface.fill(UI_COLORS['BACKGROUND'])
        
        # Draw particles
        self.simulation.particle_system.draw(self.surface)
        
        # Draw to screen
        screen.blit(self.surface, self.rect)
        
    def handle_click(self, pos):
        if not self.is_point_inside(pos):
            return False
            
        # Convert screen coordinates to simulation coordinates
        sim_pos = (
            pos[0] - self.rect.x,
            pos[1] - self.rect.y
        )
        
        if self.simulation.interaction_mode == INTERACTION_MODES['CREATE']:
            # Only create particles on initial click
            self.simulation.particle_system.create_particle_burst(
                pos=sim_pos,
                element_type=self.simulation.selected_element,
                burst_size=self.simulation.particles_per_burst,
                spread=20.0,
                speed=2.0
            )
            self.simulation.mouse_down = True
        else:  # DRAG mode
            # Convert to simulation coordinates for drag
            self.simulation.mouse_down = True
            self.simulation.particle_system.physics.start_drag(
                self.simulation.particle_system, 
                sim_pos  # Pass simulation coordinates
            )
            
        return True 