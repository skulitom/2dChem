import numpy as np
from core.constants import (
    COLLISION_RESPONSE, 
    FIXED_TIMESTEP,
    BOND_DISTANCE_THRESHOLD,
    ELECTROMAGNETIC_CONSTANT,
    COLLISION_DAMPING
)
from utils.profiler import profile_function

class CollisionHandler:
    @staticmethod
    @profile_function(threshold_ms=0.5)
    def handle_particle_collision(system, current_idx, other_idx):
        pos_diff = system.positions[current_idx] - system.positions[other_idx]
        distance = np.linalg.norm(pos_diff)
        
        # Early exit conditions
        if distance >= 3.0 or distance == 0:
            return
        
        # Calculate electromagnetic force (1/r in 2D instead of 1/r²)
        direction = pos_diff / distance
        force_magnitude = (
            ELECTROMAGNETIC_CONSTANT * 
            system.chemical_properties[current_idx].current_charge * 
            system.chemical_properties[other_idx].current_charge
        ) / distance
        
        # Apply electromagnetic force
        force = direction * force_magnitude
        system.velocities[current_idx] += force * FIXED_TIMESTEP
        system.velocities[other_idx] -= force * FIXED_TIMESTEP
        
        # Handle chemical bonding
        if distance < BOND_DISTANCE_THRESHOLD:
            current_chem = system.chemical_properties[current_idx]
            other_chem = system.chemical_properties[other_idx]
            
            # Check for bond formation based on 2D rules
            if (current_chem.can_bond_with(other_chem) and 
                CollisionHandler._check_bond_angle_validity(system, current_idx, other_idx)):
                current_chem.try_form_bond(other_chem, distance)

    @staticmethod
    @profile_function(threshold_ms=0.5)
    def _handle_regular_collision(system, idx1, idx2, pos_diff, distance):
        if distance > 0:
            direction = pos_diff / distance
            relative_velocity = system.velocities[idx1] - system.velocities[idx2]
            
            # Only apply collision response if particles are moving toward each other
            approach_speed = np.dot(relative_velocity, direction)
            if approach_speed > 0:
                # Add stronger damping to reduce bouncing
                damping = COLLISION_DAMPING
                impulse = direction * approach_speed * COLLISION_RESPONSE * damping
                
                # Apply velocity changes with mass consideration
                mass1 = system.chemical_properties[idx1].element_data.mass
                mass2 = system.chemical_properties[idx2].element_data.mass
                total_mass = mass1 + mass2
                
                # Scale impulse by mass ratio
                system.velocities[idx1] -= impulse * (mass2 / total_mass)
                system.velocities[idx2] += impulse * (mass1 / total_mass)
            
            # Separate overlapping particles more gently
            overlap = (system.chemical_properties[idx1].element_data.radius + 
                      system.chemical_properties[idx2].element_data.radius) - distance
            if overlap > 0:
                separation = direction * overlap * 0.3  # Reduce separation force
                system.positions[idx1] += separation
                system.positions[idx2] -= separation

    @staticmethod
    @profile_function(threshold_ms=0.5)
    def _handle_bonded_particles(system, idx1, idx2):
        pos_diff = system.positions[idx1] - system.positions[idx2]
        distance = np.linalg.norm(pos_diff)
        target_distance = (
            system.chemical_properties[idx1].element_data.radius +
            system.chemical_properties[idx2].element_data.radius
        ) * 0.8  # Make bonds slightly tighter
        
        if distance > 0:
            # More gentle correction for bonded particles
            correction = (distance - target_distance) * 0.1  # Reduce correction strength
            direction = pos_diff / distance
            system.positions[idx1] -= direction * correction
            system.positions[idx2] += direction * correction
            
            # Stronger velocity averaging for bonded particles
            avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
            system.velocities[idx1] = avg_velocity * 0.95  # Add slight damping
            system.velocities[idx2] = avg_velocity * 0.95

    @staticmethod
    def _check_bond_angle_validity(system, idx1, idx2) -> bool:
        """Check if new bond would satisfy 2D geometry constraints"""
        particle1 = system.chemical_properties[idx1]
        
        if len(particle1.bonds) == 0:
            return True
            
        # Get positions for angle calculations
        pos1 = system.positions[idx1]
        pos2 = system.positions[idx2]
        
        # Check angles with existing bonds
        for bond in particle1.bonds:
            pos_bonded = system.positions[bond.particle_id]
            
            # Calculate vectors
            vec1 = pos_bonded - pos1
            vec2 = pos2 - pos1
            
            # Calculate angle between bonds
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            
            # Get expected angle based on hybridization
            expected_angle = particle1.element_data.bond_angles.get(
                particle1.hybridization or 'sp2', 120.0
            )
            
            # Allow some deviation from ideal angle
            if abs(angle - expected_angle) > 15.0:  # 15 degrees tolerance
                return False
                
        return True

    @staticmethod
    def _handle_phase_changes(system, idx):
        chem = system.chemical_properties[idx]
        
        if chem.temperature >= chem.element_data.boiling_point:
            system.velocities[idx] += np.random.uniform(-1, 1, 2) * 2
            chem.break_all_bonds()
        elif chem.temperature <= chem.element_data.melting_point:
            system.velocities[idx] *= 0.8
