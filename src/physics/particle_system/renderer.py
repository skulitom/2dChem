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

        # Pre-create fonts once for reuse
        self.bond_font = pygame.font.Font(None, 24)
        self.font = pygame.font.Font(None, 24)  # For debug info or other text

        # Create element surface cache with hardware acceleration and varying sizes
        self.element_surface_cache = {}
        # Preparing element surfaces once at initialization
        for element_id, properties in ELEMENT_DATA.items():
            display_radius = int(properties.radius * 20)
            surf_size = (display_radius + 2) * 2  # +2 for outline
            surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA | pygame.HWSURFACE)

            center = (surf_size // 2, surf_size // 2)
            # Outline
            pygame.draw.circle(surf, (0, 0, 0), center, display_radius + 1)
            # Element circle
            pygame.draw.circle(surf, properties.color, center, display_radius)

            self.element_surface_cache[element_id] = {
                'surface': surf.convert_alpha(),
                'radius': display_radius + 1  # Include outline in the radius
            }

    @profile_function(threshold_ms=1.0)
    def draw(self, system, screen):
        """Draw all active particles and their bonds"""
        active_particles = system.active_particles
        if active_particles == 0:
            return

        # Clear surfaces
        self.particle_surface.fill((0, 0, 0, 0))
        self.bond_surface.fill((0, 0, 0, 0))

        # Local references for performance
        particle_blit = self.particle_surface.blit
        pos_array = system.positions
        chem_array = system.chemical_properties
        element_cache = self.element_surface_cache

        # Draw particles using cached surfaces
        for i in range(active_particles):
            pos = pos_array[i]
            element_id = chem_array[i].element_data.id

            if element_id in element_cache:
                element_data = element_cache[element_id]
                element_surf = element_data['surface']
                radius = element_data['radius']
                screen_pos = (int(pos[0] - radius), int(pos[1] - radius))
                particle_blit(element_surf, screen_pos)

        # Blit both surfaces to screen
        screen.blit(self.bond_surface, (0, 0))
        screen.blit(self.particle_surface, (0, 0))

    def _batch_draw_bonds(self, system, visible_indices):
        """Optimized bond drawing for visible particles only"""
        if visible_indices.size == 0:
            return

        # Local references for performance
        bond_surface = self.bond_surface
        particle_surface_blit = self.particle_surface.blit
        draw_line = pygame.draw.line
        pos_array = system.positions
        chem_array = system.chemical_properties
        bond_font = self.bond_font

        black = (0, 0, 0)
        white = (255, 255, 255)

        bond_lines = []
        bond_colors = []

        # Process visible particles
        for i in visible_indices:
            pos1 = pos_array[i]
            chem1 = chem_array[i]
            bonds = chem1.bonds

            # Draw bond count if any bonds present
            if bonds:
                # Minimal overhead: render once per particle with bonds
                text = bond_font.render(str(len(bonds)), True, (255, 255, 0))
                particle_surface_blit(text, (int(pos1[0] - 6), int(pos1[1] - 6)))

            # Process bonds
            for bond in bonds:
                if bond.particle_id < system.active_particles:
                    pos2 = pos_array[bond.particle_id]
                    screen_pos1 = (int(pos1[0]), int(pos1[1]))
                    screen_pos2 = (int(pos2[0]), int(pos2[1]))

                    # Check visibility before adding line
                    if self._is_bond_visible(screen_pos1, screen_pos2):
                        bond_lines.append((screen_pos1, screen_pos2))
                        bond_colors.append(white)

        # Draw all bonds with a black outline under a white center line
        if bond_lines:
            for line, color in zip(bond_lines, bond_colors):
                draw_line(bond_surface, black, line[0], line[1], 8)   # Black border
                draw_line(bond_surface, color, line[0], line[1], 4)   # White center

    @staticmethod
    def _is_bond_visible(pos1, pos2):
        """Check if any part of the bond is visible on screen"""
        # Quick bounding check ensures we only attempt to draw visible bonds
        return ((-PARTICLE_RADIUS <= max(pos1[0], pos2[0]) <= WINDOW_WIDTH + PARTICLE_RADIUS) and
                (-PARTICLE_RADIUS <= max(pos1[1], pos2[1]) <= WINDOW_HEIGHT + PARTICLE_RADIUS))

    def _draw_debug_info(self, system, screen):
        """Draw debug information for each particle"""
        # Local references
        pos_array = system.positions
        chem_array = system.chemical_properties
        font_render = self.font.render
        white = (255, 255, 255)
        yellow = (255, 255, 0)

        for i in range(system.active_particles):
            pos = pos_array[i]
            chem = chem_array[i]

            # Draw element symbol
            text = font_render(chem.element_data.id, True, white)
            screen.blit(text, (pos[0] - 10, pos[1] - 10))

            # Draw bond count info
            bond_text = font_render(f"{len(chem.bonds)}/{chem.max_bonds}", True, yellow)
            screen.blit(bond_text, (pos[0] - 10, pos[1] + 10))
