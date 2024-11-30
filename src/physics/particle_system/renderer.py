import pygame
import numpy as np
from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, 
    PARTICLE_RADIUS, ELEMENT_COLORS
)
from utils.profiler import profile_function

class ParticleRenderer:
    def __init__(self):
        # Pre-create surfaces with hardware acceleration and double buffering
        flags = pygame.SRCALPHA | pygame.HWSURFACE | pygame.DOUBLEBUF
        self.particle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), flags).convert_alpha()
        self.bond_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), flags).convert_alpha()
        
        # Create element surface cache with hardware acceleration
        self.element_surface_cache = {}
        print("\nInitializing element surfaces:")
        for element, color in ELEMENT_COLORS.items():
            print(f"Creating surface for {element} with color {color}")
            surf = pygame.Surface((PARTICLE_RADIUS * 2, PARTICLE_RADIUS * 2), pygame.SRCALPHA | pygame.HWSURFACE)
            pygame.draw.circle(surf, color, (PARTICLE_RADIUS, PARTICLE_RADIUS), PARTICLE_RADIUS)
            self.element_surface_cache[element] = surf.convert_alpha()

    @profile_function(threshold_ms=1.0)
    def draw(self, system, screen):
        """Draw all active particles and their bonds"""
        if system.active_particles == 0:
            return
        
        # Clear surfaces
        self.particle_surface.fill((0, 0, 0, 0))
        self.bond_surface.fill((0, 0, 0, 0))
        
        # Draw particles using cached surfaces
        for i in range(system.active_particles):
            pos = system.positions[i]
            chem = system.chemical_properties[i]
            element_id = chem.element_data.id
            
            # Use cached element surface
            if element_id in self.element_surface_cache:
                element_surf = self.element_surface_cache[element_id]
                screen_pos = (
                    int(pos[0] - PARTICLE_RADIUS),
                    int(pos[1] - PARTICLE_RADIUS)
                )
                self.particle_surface.blit(element_surf, screen_pos)
        
        # Blit both surfaces to screen
        screen.blit(self.bond_surface, (0, 0))
        screen.blit(self.particle_surface, (0, 0))

    def _batch_draw_bonds(self, system, visible_indices):
        """Optimized bond drawing for visible particles only"""
        if not visible_indices.size:
            return
        
        # Pre-allocate lists for better performance
        bond_lines = []
        bond_colors = []
        
        # Only process bonds for visible particles
        for i in visible_indices:
            pos1 = system.positions[i]
            chem1 = system.chemical_properties[i]
            
            # Only show bond count if bonds exist
            if len(chem1.bonds) > 0:
                font = pygame.font.Font(None, 24)
                text = font.render(str(len(chem1.bonds)), True, (255, 255, 0))
                self.particle_surface.blit(text, (int(pos1[0] - 6), int(pos1[1] - 6)))
            
            for bond in chem1.bonds:
                if bond.particle_id < system.active_particles:
                    pos2 = system.positions[bond.particle_id]
                    screen_pos1 = (int(pos1[0]), int(pos1[1]))
                    screen_pos2 = (int(pos2[0]), int(pos2[1]))
                    
                    if self._is_bond_visible(screen_pos1, screen_pos2):
                        bond_lines.append((screen_pos1, screen_pos2))
                        bond_colors.append((255, 255, 255))  # Pure white
        
        # Draw all bonds
        if bond_lines:
            for line, color in zip(bond_lines, bond_colors):
                pygame.draw.line(self.bond_surface, (0, 0, 0), line[0], line[1], 8)  # Black border
                pygame.draw.line(self.bond_surface, color, line[0], line[1], 4)      # White center

    @staticmethod
    def _is_bond_visible(pos1, pos2):
        """Check if any part of the bond is visible on screen"""
        return ((-PARTICLE_RADIUS <= max(pos1[0], pos2[0]) <= WINDOW_WIDTH + PARTICLE_RADIUS) and
                (-PARTICLE_RADIUS <= max(pos1[1], pos2[1]) <= WINDOW_HEIGHT + PARTICLE_RADIUS))