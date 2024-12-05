import pygame
from core.constants import (
    UI_COLORS, SIMULATION_X_OFFSET, SIMULATION_Y_OFFSET,
    WINDOW_WIDTH, WINDOW_HEIGHT, INTERACTION_MODES
)
from .base import UIComponent

class SimulationArea(UIComponent):
    """
    The main simulation area where particles are displayed and interacted with.
    """
    def __init__(self, simulation):
        # Define the simulation area as the part of the window not occupied by the sidebar/tabs
        rect = pygame.Rect(
            SIMULATION_X_OFFSET,
            SIMULATION_Y_OFFSET,
            WINDOW_WIDTH - SIMULATION_X_OFFSET,
            WINDOW_HEIGHT - SIMULATION_Y_OFFSET
        )
        super().__init__(simulation, rect)
        
    def draw(self, screen):
        """
        Draw the simulation area, including the particles and any background styling.
        """
        # Draw a subtle vertical gradient as the simulation background
        self._draw_gradient_background()
        
        # Draw particles
        self.simulation.particle_system.draw(self.surface)
        
        # Blit the simulation area to the screen
        screen.blit(self.surface, self.rect)
        
    def _draw_gradient_background(self):
        """
        Draw a subtle vertical gradient background for the simulation area.
        """
        self.surface.fill((0, 0, 0, 0))  # Clear with transparent
        
        gradient_surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        for y in range(self.rect.height):
            # Create a subtle gradient effect
            alpha = min(255, 180 + y // 4)
            color = (*UI_COLORS['BACKGROUND'], alpha)
            pygame.draw.line(gradient_surface, color, (0, y), (self.rect.width, y))
        
        self.surface.blit(gradient_surface, (0, 0))
        
    def handle_click(self, pos):
        """
        Handle mouse clicks within the simulation area.

        :param pos: The global mouse position at the time of the click.
        :return: True if the click was handled, False otherwise.
        """
        if not self.is_point_inside(pos):
            return False
            
        # Convert screen coordinates to simulation coordinates
        sim_pos = (
            pos[0] - self.rect.x,
            pos[1] - self.rect.y
        )
        
        # Handle interaction based on the current mode
        if self.simulation.interaction_mode == INTERACTION_MODES['CREATE']:
            # Create a burst of particles at the click location
            self.simulation.particle_system.create_particle_burst(
                pos=sim_pos,
                element_type=self.simulation.selected_element,
                burst_size=self.simulation.particles_per_burst,
                spread=20.0,
                speed=2.0
            )
            self.simulation.mouse_down = True
            return True
        elif self.simulation.interaction_mode == INTERACTION_MODES['DRAG']:
            # Begin dragging particles
            self.simulation.mouse_down = True
            self.simulation.particle_system.physics.start_drag(
                self.simulation.particle_system,
                sim_pos
            )
            return True
        
        return False
