import numpy as np
from core.constants import COLLISION_RESPONSE
from utils.profiler import profile_function

class CollisionHandler:
    @profile_function(threshold_ms=0.5)
    def handle_particle_collision(system, current_idx, other_idx):
        # Get positions and velocities using views
        pos_diff = system.positions[current_idx] - system.positions[other_idx]
        distance = np.linalg.norm(pos_diff)
        
        # Early exit conditions
        if distance >= 3.0 or distance == 0:
            return
        
        # Calculate collision response
        direction = pos_diff / distance
        rel_velocity = system.velocities[current_idx] - system.velocities[other_idx]
        approach_speed = np.dot(rel_velocity, direction)
        
        # Only process collision if particles are moving towards each other
        if approach_speed >= 0:
            return
        
        # Calculate impulse and update velocities
        impulse = direction * approach_speed * COLLISION_RESPONSE
        system.velocities[current_idx] -= impulse
        system.velocities[other_idx] += impulse
        
        # Handle overlap
        overlap = 2.0 - distance
        if overlap > 0:
            separation = direction * overlap * 0.5
            system.positions[current_idx] += separation
            system.positions[other_idx] -= separation
        
        # Handle temperature and bonding only if particles are close enough
        if distance < 2.5:
            collision_energy = 0.5 * np.sum(rel_velocity ** 2)
            temp_increase = collision_energy * 0.1
            
            current_chem = system.chemical_properties[current_idx]
            other_chem = system.chemical_properties[other_idx]
            
            current_chem.temperature += temp_increase
            other_chem.temperature += temp_increase
            
            # Check for bonding
            if (current_chem.temperature < current_chem.element_data.boiling_point and
                other_chem.temperature < other_chem.element_data.boiling_point):
                if current_chem.try_form_bond(other_chem, distance):
                    # Average velocities for bonded particles
                    avg_velocity = (system.velocities[current_idx] + system.velocities[other_idx]) * 0.5
                    system.velocities[current_idx] = avg_velocity
                    system.velocities[other_idx] = avg_velocity
    @profile_function(threshold_ms=0.5)
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

    @profile_function(threshold_ms=0.5)
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
