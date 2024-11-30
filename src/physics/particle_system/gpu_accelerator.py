from numba import cuda, float32, int32
import numpy as np
from core.constants import GRAVITY, COLLISION_RESPONSE, MAX_VELOCITY
import logging

# Configure all CUDA-related logging to only show errors
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.ERROR)
logging.getLogger('numba.cuda.cudadrv.runtime').setLevel(logging.ERROR)
logging.getLogger('numba.cuda.cudadrv.nvvm').setLevel(logging.ERROR)
logging.getLogger('numba.cuda.cudadrv.cuda').setLevel(logging.ERROR)

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
        velocities[idx, 1] += GRAVITY * dt
        
        # Clamp velocity
        if velocities[idx, 1] > MAX_VELOCITY:
            velocities[idx, 1] = MAX_VELOCITY
        elif velocities[idx, 1] < -MAX_VELOCITY:
            velocities[idx, 1] = -MAX_VELOCITY

@cuda.jit
def handle_collisions_gpu(positions, velocities, active_particles):
    idx = cuda.grid(1)
    if idx >= active_particles:
        return
        
    # Shared memory for position data
    pos_shared = cuda.shared.array(shape=(256, 2), dtype=float32)
    vel_shared = cuda.shared.array(shape=(256, 2), dtype=float32)
    
    # Load position and velocity into shared memory
    tx = cuda.threadIdx.x
    pos_shared[tx] = positions[idx]
    vel_shared[tx] = velocities[idx]
    cuda.syncthreads()
    
    for j in range(active_particles):
        if idx != j:
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
                    impulse = vn * COLLISION_RESPONSE
                    vel_shared[tx][0] -= impulse * nx
                    vel_shared[tx][1] -= impulse * ny
    
    # Write back to global memory
    velocities[idx] = vel_shared[tx] 