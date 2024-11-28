import pygame
import numpy as np
from core.constants import WINDOW_WIDTH, WINDOW_HEIGHT, ELEMENT_COLORS

class ParticleRenderer:
    @staticmethod
    def draw(system, screen):
        if system.active_particles == 0:
            return
        
        positions = system.positions[:system.active_particles].astype(np.int32)
        particle_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        
        ParticleRenderer._draw_particles(system, particle_surface, positions)
        ParticleRenderer._draw_bonds(system, screen, positions)
        
        screen.blit(particle_surface, (0, 0))

    @staticmethod
    def _draw_particles(system, surface, positions):
        for element in set(system.element_types[:system.active_particles]):
            element_mask = system.element_types[:system.active_particles] == element
            element_positions = positions[element_mask]
            
            element_color = ELEMENT_COLORS[element]
            base_color = np.array([element_color.r, element_color.g, element_color.b])
            
            temps = np.array([system.chemical_properties[i].temperature 
                            for i in range(system.active_particles) 
                            if system.element_types[i] == element])
            
            temp_factors = np.clip((temps - 273.15) / 1000, 0, 1)
            hot_color = np.array([255, 200, 0])
            
            colors = np.array([tuple((base_color * (1 - tf) + hot_color * tf).astype(int))
                             for tf in temp_factors])
            
            for pos, color in zip(element_positions, colors):
                pygame.draw.circle(surface, color, pos, 2)

    @staticmethod
    def _draw_bonds(system, screen, positions):
        bond_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        processed_bonds = set()
        
        for i in range(system.active_particles):
            pos1 = positions[i]
            for bond in system.chemical_properties[i].bonds:
                bond_key = tuple(sorted([i, bond.particle_id]))
                if bond_key not in processed_bonds and bond.particle_id < system.active_particles:
                    processed_bonds.add(bond_key)
                    pos2 = positions[bond.particle_id]
                    
                    bond_color = (155 + int(bond.strength * 40),) * 3 if bond.bond_type == 'covalent' else (100, 100, 255)
                    pygame.draw.line(bond_surface, bond_color, pos1, pos2, 1)
        
        screen.blit(bond_surface, (0, 0)) 