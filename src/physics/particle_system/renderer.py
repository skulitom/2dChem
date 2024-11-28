import pygame
import numpy as np
from core.constants import WINDOW_WIDTH, WINDOW_HEIGHT, ELEMENT_COLORS
from utils.profiler import profile_function

class ParticleRenderer:
    @profile_function(threshold_ms=1.0)
    def draw(self, system, screen):
        if system.active_particles == 0:
            return
        
        positions = system.positions[:system.active_particles].astype(np.int32)
        
        # Create a new surface for this frame
        particle_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        
        # Draw particles and bonds
        self._draw_particles(system, particle_surface, positions)
        self._draw_bonds(system, particle_surface, positions)  # Changed to draw on particle_surface
        
        # Blit the complete surface to the screen
        screen.blit(particle_surface, (0, 0))

    @profile_function(threshold_ms=0.5)
    def _draw_particles(self, system, surface, positions):
        # Pre-allocate arrays for better performance
        active_elements = set(system.element_types[:system.active_particles])
        
        for element in active_elements:
            # Get all particles of this element type at once
            element_mask = system.element_types[:system.active_particles] == element
            element_positions = positions[element_mask]
            
            # Get temperatures for all particles of this element at once
            element_indices = np.where(element_mask)[0]
            temps = np.array([system.chemical_properties[i].temperature 
                             for i in element_indices])
            
            # Vectorized color calculations
            element_color = np.array(ELEMENT_COLORS[element][:3])
            hot_color = np.array([255, 200, 0])
            temp_factors = np.clip((temps - 273.15) / 1000, 0, 1)
            
            # Vectorized color blending
            colors = np.array([
                tuple((element_color * (1 - tf) + hot_color * tf).astype(int))
                for tf in temp_factors
            ])
            
            # Draw particles using pygame.draw.rect for reliable rendering
            for pos, color in zip(element_positions, colors):
                x, y = pos.astype(int)
                if 0 <= x < surface.get_width() - 2 and 0 <= y < surface.get_height() - 2:
                    pygame.draw.rect(surface, color, (x, y, 2, 2))

    @profile_function(threshold_ms=0.5)
    def _draw_bonds(self, system, screen, positions):
        # Pre-allocate bond surface
        bond_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        processed_bonds = set()
        
        # Create lists for batch drawing
        lines = []
        colors = []
        
        for i in range(system.active_particles):
            pos1 = positions[i]
            for bond in system.chemical_properties[i].bonds:
                bond_key = tuple(sorted([i, bond.particle_id]))
                if bond_key not in processed_bonds and bond.particle_id < system.active_particles:
                    processed_bonds.add(bond_key)
                    pos2 = positions[bond.particle_id]
                    
                    lines.append((pos1, pos2))
                    colors.append((155 + int(bond.strength * 40),) * 3 if bond.bond_type == 'covalent' else (100, 100, 255))
        
        # Batch draw lines
        if lines:
            pygame.draw.lines(bond_surface, (255, 255, 255), False, lines[0], 1)
            for line, color in zip(lines, colors):
                pygame.draw.line(bond_surface, color, line[0], line[1], 1)
        
        screen.blit(bond_surface, (0, 0)) 