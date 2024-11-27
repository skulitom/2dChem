import numpy as np

SIMULATION_STEP = 0.1

def solve_velocities(velocities):
    velocities[:, 1] += 1 * SIMULATION_STEP

def solve_positions(velocities, positions):
    positions[:, 1] += velocities[:, 1] * SIMULATION_STEP
    positions[:, 2] += velocities[:, 2] * SIMULATION_STEP
