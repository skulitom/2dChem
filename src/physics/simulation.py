import numpy as np

from physics.particle_data import ParticleData

from core.constants import (
    SIMULATION_WIDTH, SIMULATION_HEIGHT
)
SIMULATION_STEP = 0.1

def _solve_velocities(particle_data : ParticleData):
    particle_data.velocities[:, 1] += 9.8 * SIMULATION_STEP

def _solve_positions(particle_data : ParticleData):
    particle_data.positions[:, 0] += particle_data.velocities[:, 0] * SIMULATION_STEP
    particle_data.positions[:, 1] += particle_data.velocities[:, 1] * SIMULATION_STEP

def _solve_borders_constrains(particle_data : ParticleData):
    active_mask = np.arange(particle_data.num_of_particles)
    bottom_mask = active_mask[particle_data.positions[active_mask, 1] >= SIMULATION_HEIGHT]

    particle_data.positions[bottom_mask, 1] = SIMULATION_HEIGHT - 1
    particle_data.velocities[bottom_mask, 1] = 0

    left_mask = active_mask[particle_data.positions[active_mask, 0] <= 0]
    particle_data.positions[left_mask, 0] = 1
    particle_data.velocities[left_mask, 0] = 0

    right_mask = active_mask[particle_data.positions[active_mask, 0] >= SIMULATION_WIDTH]
    particle_data.positions[right_mask, 0] = SIMULATION_WIDTH - 1
    particle_data.velocities[right_mask, 0] = 0

#def _populate_grid(particle_data : ParticleData):
#    for id, position in np.

def _solve_self_collisions(particle_data : ParticleData):
    collisions_mask = np.empty((0, 2), dtype=int)
    for left_el_id in range(0, particle_data.num_of_particles):
        for right_el_id in range(left_el_id + 1, particle_data.num_of_particles):
            left_position = particle_data.positions[left_el_id]
            right_position = particle_data.positions[right_el_id]
            # Compute the distance
            direction = left_position - right_position
            distance = np.linalg.norm(direction)

            if distance <= 0.01:
                distance = 0.01

            # Check if particles collide
            if distance < particle_data.radius * 2:
                # Add the pair of indices to the collisions_mask
                collisions_mask = np.append(collisions_mask, [[left_el_id, right_el_id]], axis=0)
                mid_point = (left_position + right_position) / 2
                vector = direction / distance

                particle_data.positions[left_el_id] = mid_point + vector * particle_data.radius
                particle_data.positions[right_el_id] = mid_point - vector * particle_data.radius

def solve(particle_data_list : list):
    pass
#    for particle_data in particle_data_list:
#        _solve_velocities(particle_data)
#        _solve_positions(particle_data)
#        _solve_self_collisions(particle_data)
#        _solve_borders_constrains(particle_data)

