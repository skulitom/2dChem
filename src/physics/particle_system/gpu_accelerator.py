from numba import cuda, float32, int32
import numpy as np
from core.constants import (
    SIMULATION_X_OFFSET, SIMULATION_WIDTH, PARTICLE_RADIUS
)
import logging

# Configure all CUDA-related logging to only show errors
cuda_loggers = [
    'numba',
    'numba.cuda.cudadrv.driver',
    'numba.cuda.cudadrv.runtime',
    'numba.cuda.cudadrv.nvvm',
    'numba.cuda.cudadrv.cuda'
]
for logger_name in cuda_loggers:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# CUDA constants need to be defined at module level
CUDA_MAX_VELOCITY = 20.0
CUDA_GRAVITY = 4.0
CUDA_COLLISION_RESPONSE = 0.3
CUDA_SIMULATION_X_OFFSET = float32(SIMULATION_X_OFFSET)
CUDA_SIMULATION_WIDTH = float32(SIMULATION_WIDTH)
CUDA_PARTICLE_RADIUS = float32(PARTICLE_RADIUS)

@cuda.jit
def update_positions_gpu(positions, velocities, dt):
    idx = cuda.grid(1)
    if idx < positions.shape[0]:
        # Update position
        positions[idx, 0] += velocities[idx, 0] * dt
        positions[idx, 1] += velocities[idx, 1] * dt

@cuda.jit
def update_velocities_gpu(velocities, dt):
    idx = cuda.grid(1)
    if idx < velocities.shape[0]:
        # Apply gravity
        velocities[idx, 1] += CUDA_GRAVITY * dt
        
        # Clamp velocity
        if velocities[idx, 1] > CUDA_MAX_VELOCITY:
            velocities[idx, 1] = CUDA_MAX_VELOCITY
        elif velocities[idx, 1] < -CUDA_MAX_VELOCITY:
            velocities[idx, 1] = -CUDA_MAX_VELOCITY

@cuda.jit
def handle_collisions_gpu(positions, velocities, active_particles):
    # Add synchronization to prevent race conditions
    cuda.syncthreads()  # Synchronize before processing
    
    particle_idx = cuda.grid(1)
    if particle_idx >= active_particles:
        return

    # Load position and velocity
    pos_x = positions[particle_idx, 0]
    pos_y = positions[particle_idx, 1]
    vel_x = velocities[particle_idx, 0]
    vel_y = velocities[particle_idx, 1]
    
    # Handle boundaries - positions are relative to simulation area
    if pos_x < 0:
        positions[particle_idx, 0] = 0
        velocities[particle_idx, 0] *= -CUDA_COLLISION_RESPONSE
    elif pos_x > CUDA_SIMULATION_WIDTH - CUDA_PARTICLE_RADIUS:
        positions[particle_idx, 0] = CUDA_SIMULATION_WIDTH - CUDA_PARTICLE_RADIUS
        velocities[particle_idx, 0] *= -CUDA_COLLISION_RESPONSE
    
    # Shared memory for position data
    tx = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    pos_shared = cuda.shared.array(shape=(block_size, 2), dtype=float32)
    vel_shared = cuda.shared.array(shape=(block_size, 2), dtype=float32)
    
    # Load position and velocity into shared memory
    if particle_idx < positions.shape[0]:
        pos_shared[tx] = positions[particle_idx]
        vel_shared[tx] = velocities[particle_idx]
    cuda.syncthreads()
    
    # Handle particle collisions
    for j in range(active_particles):
        if j >= positions.shape[0]:
            break
        if particle_idx != j:
            dx = pos_shared[tx][0] - positions[j][0]
            dy = pos_shared[tx][1] - positions[j][1]
            dist_sq = dx * dx + dy * dy
            
            if dist_sq < 9.0 and dist_sq > 0:  # 3.0 * 3.0
                dist = cuda.math.sqrt(dist_sq)
                nx = dx / dist
                ny = dy / dist
                
                # Calculate relative velocity
                dvx = vel_shared[tx][0] - velocities[j][0]
                dvy = vel_shared[tx][1] - velocities[j][1]
                
                # Normal velocity component
                vn = dvx * nx + dvy * ny
                
                if vn > 0:
                    # Collision response
                    impulse = vn * CUDA_COLLISION_RESPONSE
                    vel_shared[tx][0] -= impulse * nx
                    vel_shared[tx][1] -= impulse * ny
    
    # Ensure all threads complete before writing back
    cuda.syncthreads()
    # Write back to global memory
    velocities[particle_idx] = vel_shared[tx]