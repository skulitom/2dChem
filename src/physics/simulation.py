import numpy as np

from particle_data import ParticleData

SIMULATION_STEP = 0.1

def _solve_velocities(particle_data : ParticleData):
    particle_data.velocities[:, 1] += 1 * SIMULATION_STEP

def _solve_positions(particle_data : ParticleData):
    particle_data.positions[:, 1] += particle_data.velocities[:, 1] * SIMULATION_STEP
    particle_data.positions[:, 2] += particle_data.velocities[:, 2] * SIMULATION_STEP

def solve(particle_data_list : list):
    for particle_data in particle_data_list:
        _solve_velocities(particle_data)
        _solve_positions(particle_data)
