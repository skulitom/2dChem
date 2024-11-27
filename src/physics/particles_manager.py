import numpy as np



def create_partices(velocities, positions, coord):
    offsets = np.random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD, (new_count, 2))
    new_positions = np.tile(pos, (new_count, 1)) + offsets
    new_velocities = np.random.uniform(-0.5, 0.5, (new_count, 2)) * VELOCITY_SCALE
