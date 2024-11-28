import numpy as np
from pygame import Color
import pygame

from physics.particle_data import ParticleData

from core.constants import (
    PARTICLE_SPREAD, VELOCITY_SCALE, SIMULATION_FRAME_WIDTH, SIMULATION_FRAME_HEIGHT
)

def create_partices(particle_data : ParticleData, coord):
    new_count = 5
    offsets = np.random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD, (new_count, 2))
    new_positions = np.tile(coord, (new_count, 1)) + offsets
    new_velocities = np.random.uniform(-0.5, 0.5, (new_count, 2)) * VELOCITY_SCALE

    new_positions[:, 0] = np.clip(new_positions[:, 0], 0, SIMULATION_FRAME_WIDTH - 1)
    new_positions[:, 1] = np.clip(new_positions[:, 1], 0, SIMULATION_FRAME_HEIGHT - 1)
    
    print("num_of_elements", particle_data.num_of_particles)
    start_idx = particle_data.num_of_particles
    end_idx = start_idx + new_count
    particle_data.positions[start_idx:end_idx] = new_positions
    particle_data.velocities[start_idx:end_idx] = new_velocities

    particle_data.num_of_particles = particle_data.num_of_particles + new_count

def render_particles(screen, particle_data : ParticleData):
    for position in particle_data.positions:
        pos = position.astype(np.int32)
        pygame.draw.circle(screen, particle_data.color, pos, particle_data.radius)

def render_grid(screen):
    pygame.draw.line(screen, Color(255, 0, 255), (0, 0), (0, 100), 10)
