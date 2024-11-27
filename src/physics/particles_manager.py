import numpy as np
from pygame import Color

from particle_data import ParticleData

from core.constants import (
    PARTICLE_SPREAD, VELOCITY_SCALE, WINDOW_WIDTH, WINDOW_HEIGHT
)

def create_partices(particle_data : ParticleData, coord):
    new_count = 5
    offsets = np.random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD, (new_count, 2))
    new_positions = np.tile(coord, (new_count, 1)) + offsets
    new_velocities = np.random.uniform(-0.5, 0.5, (new_count, 2)) * VELOCITY_SCALE

    new_positions[:, 0] = np.clip(new_positions[:, 0], 0, WINDOW_WIDTH - 1)
    new_positions[:, 1] = np.clip(new_positions[:, 1], 0, WINDOW_HEIGHT - 1)
    
    start_idx = len(particle_data.positions)
    end_idx = start_idx + new_count
    particle_data.positions[start_idx:end_idx] = new_positions
    particle_data.velocities[start_idx:end_idx] = new_velocities

def render_particles(screen, particle_data : ParticleData):
    for i in range(particle_data.positions):
        pos = particle_data.positions[i].astype(np.int32)
        screen.set_at(pos, particle_data.color)
