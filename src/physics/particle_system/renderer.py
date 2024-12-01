import pygame
import numpy as np
from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, 
    PARTICLE_RADIUS
)
from core.element_data import ELEMENT_DATA
from utils.profiler import profile_function

class ParticleRenderer:
    def __init__(self):
        # Pre-create surfaces with hardware acceleration and double buffering
        flags = pygame.SRCALPHA | pygame.HWSURFACE | pygame.DOUBLEBUF
        self.particle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), flags).convert_alpha()
        self.bond_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), flags).convert_alpha()
        
        # Create element surface cache with hardware acceleration and varying sizes
        self.element_surface_cache = {}
        print("\nInitializing element surfaces:")
        for element_id, properties in ELEMENT_DATA.items():
            # Scale radius based on element's atomic radius (reduced scaling factor)
            display_radius = int(properties.radius * 20)  # Reduced from 40 to 20
            print(f"Creating surface for {element_id} with radius {display_radius}")
            
            # Create surface large enough for the element plus outline
            surf_size = (display_radius + 2) * 2  # +2 for outline
            surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA | pygame.HWSURFACE)
            
            # Draw outline (black circle slightly larger than the element)
            center = (surf_size // 2, surf_size // 2)
            pygame.draw.circle(surf, (0, 0, 0), center, display_radius + 1)
            
            # Draw the element circle
            color = properties.color
            pygame.draw.circle(surf, color, center, display_radius)
            
            # Store both the surface and the radius for later use
            self.element_surface_cache[element_id] = {
                'surface': surf.convert_alpha(),
                'radius': display_radius + 1  # Include outline in radius
            }

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
                element_data = self.element_surface_cache[element_id]
                element_surf = element_data['surface']
                radius = element_data['radius']
                
                screen_pos = (
                    int(pos[0] - radius),
                    int(pos[1] - radius)
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

    def _draw_debug_info(self, system, screen):
        """Draw debug information"""
        for i in range(system.active_particles):
            pos = system.positions[i]
            chem = system.chemical_properties[i]
            
            # Draw element symbol
            text = self.font.render(chem.element_data.id, True, (255, 255, 255))
            screen.blit(text, (pos[0] - 10, pos[1] - 10))
            
            # Draw bond count
            bond_text = self.font.render(f"{len(chem.bonds)}/{chem.max_bonds}", True, (255, 255, 0))
            screen.blit(bond_text, (pos[0] - 10, pos[1] + 10))