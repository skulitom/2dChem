import numpy as np
from core.constants import (
    COLLISION_RESPONSE, 
    FIXED_TIMESTEP,
    BOND_DISTANCE_THRESHOLD,
    ELECTROMAGNETIC_CONSTANT,
    COLLISION_DAMPING,
    ACTIVATION_ENERGY_THRESHOLD
)
from utils.profiler import profile_function
from physics.chemical_particle import Bond
from typing import Optional

class CollisionHandler:
    @staticmethod
    @profile_function(threshold_ms=0.5)
    def handle_particle_collision(system, current_idx, other_idx):
        pos_diff = system.positions[current_idx] - system.positions[other_idx]
        distance = np.linalg.norm(pos_diff)
        
        if distance >= 3.0 or distance == 0:
            return
            
        current_chem = system.chemical_properties[current_idx]
        other_chem = system.chemical_properties[other_idx]
        
        # Calculate collision energy
        relative_velocity = system.velocities[current_idx] - system.velocities[other_idx]
        collision_energy = 0.5 * np.linalg.norm(relative_velocity) ** 2
        
        # Handle different types of interactions
        if current_chem.can_form_ionic_bond(other_chem):
            CollisionHandler._handle_ionic_interaction(system, current_idx, other_idx, collision_energy)
        elif current_chem.can_bond_with(other_chem, distance):
            CollisionHandler._handle_covalent_interaction(system, current_idx, other_idx, distance, collision_energy)
        else:
            CollisionHandler._handle_regular_collision(system, current_idx, other_idx, pos_diff, distance)

    @staticmethod
    def _handle_ionic_interaction(system, idx1, idx2, collision_energy):
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        # Check if energy is sufficient for electron transfer
        if collision_energy > ACTIVATION_ENERGY_THRESHOLD:
            # Transfer electron from lower to higher electronegativity
            if chem1.element_data.electronegativity > chem2.element_data.electronegativity:
                donor, acceptor = chem2, chem1
            else:
                donor, acceptor = chem1, chem2
                
            donor.current_charge += 1
            acceptor.current_charge -= 1
            donor.valence_electrons -= 1
            acceptor.valence_electrons += 1

    @staticmethod
    def _handle_covalent_interaction(system, idx1, idx2, distance, collision_energy):
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        if collision_energy > ACTIVATION_ENERGY_THRESHOLD * 0.5:  # Lower threshold for covalent bonds
            # Check bond angle constraints
            angle = CollisionHandler._calculate_bond_angle(system, idx1, idx2)
            if angle is not None:
                # Get expected angle based on hybridization
                expected_angle = chem1.preferred_geometry.get('trigonal', 120.0)
                if abs(angle - expected_angle) > 30.0:  # Allow 30-degree deviation
                    return  # Angle not suitable for bonding
            
            # Determine number of electrons to share based on valence
            shared_electrons = min(
                2,  # Start with single bonds for simplicity
                chem1.valence_electrons,
                chem2.valence_electrons
            )
            
            if shared_electrons > 0:
                # Create bonds in both directions
                bond1 = Bond(
                    particle_id=idx2,
                    bond_type='covalent',
                    strength=1.0,
                    shared_electrons=shared_electrons,
                    angle=angle
                )
                bond2 = Bond(
                    particle_id=idx1,
                    bond_type='covalent',
                    strength=1.0,
                    shared_electrons=shared_electrons,
                    angle=None  # Only store angle on one side
                )
                
                chem1.bonds.append(bond1)
                chem2.bonds.append(bond2)
                
                # Update velocities to reflect bond formation
                avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
                system.velocities[idx1] = avg_velocity * 0.9  # Add some damping
                system.velocities[idx2] = avg_velocity * 0.9

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

    @staticmethod
    def _calculate_bond_angle(system, idx1, idx2) -> Optional[float]:
        """Calculate the angle between existing bonds and potential new bond"""
        particle1 = system.chemical_properties[idx1]
        
        if len(particle1.bonds) == 0:
            return None  # No existing bonds to form angle with
            
        pos1 = system.positions[idx1]
        pos2 = system.positions[idx2]
        
        # If there's one existing bond, calculate angle with it
        if len(particle1.bonds) == 1:
            existing_bond = particle1.bonds[0]
            pos_existing = system.positions[existing_bond.particle_id]
            
            # Calculate vectors
            vec1 = pos_existing - pos1
            vec2 = pos2 - pos1
            
            # Calculate angle between vectors
            dot_product = np.dot(vec1, vec2)
            magnitudes = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            if magnitudes == 0:
                return None
                
            cos_angle = dot_product / magnitudes
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            return angle
            
        # If there are two existing bonds, calculate both angles
        elif len(particle1.bonds) == 2:
            angles = []
            for bond in particle1.bonds:
                pos_existing = system.positions[bond.particle_id]
                vec1 = pos_existing - pos1
                vec2 = pos2 - pos1
                
                dot_product = np.dot(vec1, vec2)
                magnitudes = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                
                if magnitudes == 0:
                    continue
                    
                cos_angle = dot_product / magnitudes
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                angles.append(angle)
                
            return min(angles) if angles else None
            
        return None
