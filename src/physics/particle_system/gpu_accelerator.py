from numba import cuda, float32, int32
import math
import logging
import numpy as np
from core.constants import (
    SIMULATION_WIDTH, SIMULATION_HEIGHT, PARTICLE_RADIUS
)

# Configure CUDA logging to errors only
cuda_loggers = [
    'numba',
    'numba.cuda.cudadrv.driver',
    'numba.cuda.cudadrv.runtime',
    'numba.cuda.cudadrv.nvvm',
    'numba.cuda.cudadrv.cuda'
]
for logger_name in cuda_loggers:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

CUDA_MAX_VELOCITY = 20.0
CUDA_GRAVITY = 4.0
CUDA_COLLISION_RESPONSE = 1.0
COLLISION_DIST_FACTOR = 1.1

@cuda.jit
def update_positions_gpu(positions, velocities, dt):
    idx = cuda.grid(1)
    if idx < positions.shape[0]:
        positions[idx, 0] += velocities[idx, 0] * dt
        positions[idx, 1] += velocities[idx, 1] * dt

@cuda.jit
def update_velocities_gpu(velocities, dt):
    idx = cuda.grid(1)
    if idx < velocities.shape[0]:
        vy = velocities[idx, 1] + CUDA_GRAVITY * dt
        # Clamp velocities
        vy = min(max(vy, -CUDA_MAX_VELOCITY), CUDA_MAX_VELOCITY)
        vx = min(max(velocities[idx, 0], -CUDA_MAX_VELOCITY), CUDA_MAX_VELOCITY)
        velocities[idx, 0] = vx
        velocities[idx, 1] = vy

@cuda.jit
def handle_collisions_gpu(positions, velocities, radii, active_particles,
                          particle_indices, cell_start, cell_end,
                          grid_cells_x, grid_cells_y, cell_size_x, cell_size_y):
    """Handle collisions using spatial hashing.

    Parameters:
    - positions, velocities, radii: particle data
    - active_particles: number of active particles
    - particle_indices: array mapping sorted particles to original indices
    - cell_start, cell_end: prefix arrays giving start/end of particle indices for each cell
    - grid_cells_x, grid_cells_y: number of cells in x and y direction
    - cell_size_x, cell_size_y: size of each cell

    Each particle:
    1. Finds which cell it's in.
    2. Checks its own cell and the 8 neighboring cells.
    3. Only computes collisions with particles in those cells.
    """
    idx = cuda.grid(1)
    if idx >= active_particles:
        return

    # Original particle index after sorting
    orig_idx = particle_indices[idx]

    # Load this particle's data
    pos_x = positions[orig_idx, 0]
    pos_y = positions[orig_idx, 1]
    vel_x = velocities[orig_idx, 0]
    vel_y = velocities[orig_idx, 1]
    my_radius = radii[orig_idx]

    # Compute the cell coordinates for this particle
    cell_x = int(pos_x / cell_size_x)
    cell_y = int(pos_y / cell_size_y)

    # Collision distance factor (slightly larger to ensure no overlap)
    # Precompute squared radius for early-out checks
    # We'll check up to 9 cells: current + 8 neighbors
    for neighbor_y in range(max(cell_y - 1, 0), min(cell_y + 2, grid_cells_y)):
        for neighbor_x in range(max(cell_x - 1, 0), min(cell_x + 2, grid_cells_x)):
            cell_idx = neighbor_y * grid_cells_x + neighbor_x
            start = cell_start[cell_idx]
            end = cell_end[cell_idx]

            if start == -1:
                # Empty cell
                continue

            # Iterate over particles in the neighboring cell
            for k in range(start, end):
                other_orig_idx = particle_indices[k]
                if other_orig_idx == orig_idx:
                    continue

                dx = pos_x - positions[other_orig_idx, 0]
                # Early out check on dx
                if dx > 3.0 * 2.0 or dx < -3.0 * 2.0:
                    continue
                dy = pos_y - positions[other_orig_idx, 1]

                # Quick rejection if too far in either axis
                if (dy > 3.0 * 2.0 or dy < -3.0 * 2.0):
                    continue

                dist_sq = dx * dx + dy * dy
                other_radius = radii[other_orig_idx]
                min_dist = (my_radius + other_radius) * COLLISION_DIST_FACTOR

                if dist_sq < min_dist * min_dist and dist_sq > 0.0:
                    dist = math.sqrt(dist_sq)
                    nx = dx / dist
                    ny = dy / dist

                    # Position correction
                    overlap = min_dist - dist
                    pos_x += nx * overlap * 0.5
                    pos_y += ny * overlap * 0.5

                    # Swap velocities along normal
                    other_vel_x = velocities[other_orig_idx, 0]
                    other_vel_y = velocities[other_orig_idx, 1]

                    # Simple elastic collision response
                    temp_vel_x = vel_x
                    temp_vel_y = vel_y
                    vel_x = other_vel_x
                    vel_y = other_vel_y
                    other_vel_x = temp_vel_x
                    other_vel_y = temp_vel_y

                    velocities[other_orig_idx, 0] = other_vel_x
                    velocities[other_orig_idx, 1] = other_vel_y

    # Write back updated position and velocity
    positions[orig_idx, 0] = pos_x
    positions[orig_idx, 1] = pos_y
    velocities[orig_idx, 0] = vel_x
    velocities[orig_idx, 1] = vel_y
