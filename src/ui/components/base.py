import pygame
from abc import ABC, abstractmethod

class UIComponent(ABC):
    def __init__(self, simulation, rect):
        self.simulation = simulation
        self.rect = rect
        self.surface = pygame.Surface(rect.size)
    
    @abstractmethod
    def draw(self, screen):
        """Draw the component to the screen"""
        pass
        
    @abstractmethod
    def handle_click(self, pos):
        """Handle mouse click events"""
        pass
        
    def is_point_inside(self, pos):
        """Check if point is inside component"""
        return self.rect.collidepoint(pos) 