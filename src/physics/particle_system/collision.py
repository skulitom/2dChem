import numpy as np
from core.constants import COLLISION_RESPONSE

class CollisionHandler:
    @staticmethod
    def handle_particle_collision(system, current_idx, other_idx):
        current_chem = system.chemical_properties[current_idx]
        other_chem = system.chemical_properties[other_idx]
        
        rel_velocity = np.linalg.norm(system.velocities[current_idx] - system.velocities[other_idx])
        collision_energy = 0.5 * (current_chem.element_data.mass * rel_velocity ** 2)
        
        temp_increase = collision_energy * 0.1
        current_chem.temperature += temp_increase
        other_chem.temperature += temp_increase
        
        CollisionHandler._handle_phase_changes(system, current_idx)
        CollisionHandler._handle_phase_changes(system, other_idx)
        
        pos_diff = system.positions[current_idx] - system.positions[other_idx]
        distance = np.linalg.norm(pos_diff)
        
        if (current_chem.temperature < current_chem.element_data.boiling_point and
            other_chem.temperature < other_chem.element_data.boiling_point):
            if current_chem.try_form_bond(other_chem, distance):
                CollisionHandler._handle_bonded_particles(system, current_idx, other_idx)
        else:
            CollisionHandler._handle_regular_collision(system, current_idx, other_idx, pos_diff, distance)

    @staticmethod
    def _handle_regular_collision(system, idx1, idx2, pos_diff, distance):
        if distance > 0:
            direction = pos_diff / distance
            relative_velocity = system.velocities[idx1] - system.velocities[idx2]
            
            impulse = direction * np.dot(relative_velocity, direction) * COLLISION_RESPONSE
            
            system.velocities[idx1] -= impulse
            system.velocities[idx2] += impulse
            
            overlap = 2 - distance
            if overlap > 0:
                separation = direction * overlap * 0.5
                system.positions[idx1] += separation
                system.positions[idx2] -= separation

    @staticmethod
    def _handle_bonded_particles(system, idx1, idx2):
        pos_diff = system.positions[idx1] - system.positions[idx2]
        distance = np.linalg.norm(pos_diff)
        target_distance = (system.chemical_properties[idx1].element_data.radius +
                         system.chemical_properties[idx2].element_data.radius)
        
        if distance > 0:
            correction = (distance - target_distance) * 0.5
            direction = pos_diff / distance
            system.positions[idx1] -= direction * correction
            system.positions[idx2] += direction * correction
            
            avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
            system.velocities[idx1] = avg_velocity
            system.velocities[idx2] = avg_velocity

    @staticmethod
    def _handle_phase_changes(system, idx):
        chem = system.chemical_properties[idx]
        
        if chem.temperature >= chem.element_data.boiling_point:
            system.velocities[idx] += np.random.uniform(-1, 1, 2) * 2
            chem.break_all_bonds()
        elif chem.temperature <= chem.element_data.melting_point:
            system.velocities[idx] *= 0.8 