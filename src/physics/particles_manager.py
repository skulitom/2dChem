import numpy as np
from pygame import Color
import pygame

from physics.particle_data import ParticleData

from core.constants import (
    PARTICLE_SPREAD, VELOCITY_SCALE, SIMULATION_WIDTH, SIMULATION_HEIGHT
)

def compute_grid_cell_id(particle_data : ParticleData, coord):
    x_id = int(coord[0] // particle_data.grid_cell_size)
    y_id = int(coord[1] // particle_data.grid_cell_size)

    id = x_id + (y_id * particle_data.grid_cells_per_row)

    return id 

def create_partices(particle_data : ParticleData, coord):
    new_count = 5
    offsets = np.random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD, (new_count, 2))
    new_positions = np.tile(coord, (new_count, 1)) + offsets
    new_velocities = np.random.uniform(-0.5, 0.5, (new_count, 2)) * VELOCITY_SCALE

    new_positions[:, 0] = np.clip(new_positions[:, 0], 0, SIMULATION_WIDTH - 1)
    new_positions[:, 1] = np.clip(new_positions[:, 1], 0, SIMULATION_HEIGHT - 1)

    for position in new_positions:
        cell_id = compute_grid_cell_id(particle_data, position)
        cell = particle_data.grid[cell_id]
        
        start_idx = cell.num_of_particles
        end_idx = start_idx + new_count
        cell.positions[start_idx:end_idx] = new_positions
        cell.velocities[start_idx:end_idx] = new_velocities

        cell.num_of_particles = cell.num_of_particles + new_count

def render_particles(screen, particle_data : ParticleData):
    for cell in particle_data.grid:
        for el_id in range(0, cell.num_of_particles):
            position = cell.positions[el_id]
            pos = position.astype(np.int32)
            pygame.draw.circle(screen, particle_data.color, pos, particle_data.radius)

def render_grid(screen, particle_data : ParticleData):
    def draw_cell(screen, cell_id, cells_per_row, cell_width):
        x = cell_width * (cell_id % cells_per_row)
        y = cell_width * (cell_id // cells_per_row)
        pygame.draw.line(screen, Color(255, 0, 255), (x, y), (x, y + cell_width), 5)
        pygame.draw.line(screen, Color(255, 0, 255), (x, y), (x + cell_width, y), 5)
        pygame.draw.line(screen, Color(255, 0, 255), (x + cell_width, y), (x + cell_width, y + cell_width), 5)
        pygame.draw.line(screen, Color(255, 0, 255), (x, y + cell_width), (x + cell_width, y + cell_width), 5)
    
    for id in range(0, particle_data.grid_num_of_cells):
        draw_cell(screen, id, particle_data.grid_cells_per_row, particle_data.grid_cell_size)
