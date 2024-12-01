import numpy as np
from core.constants import (
    COLLISION_RESPONSE, 
    FIXED_TIMESTEP,
    BOND_DISTANCE_THRESHOLD,
    ELECTROMAGNETIC_CONSTANT,
    COLLISION_DAMPING,
    ACTIVATION_ENERGY_THRESHOLD,
    BOND_ANGLE_TOLERANCE
)
from utils.profiler import profile_function
from physics.chemical_particle import Bond
from typing import Optional
from physics.bond_system import BondSystem
class CollisionHandler:
    @staticmethod
    @profile_function(threshold_ms=0.5)
    def handle_particle_collision(system, current_idx, other_idx):
        """Handle collision between two particles"""
        # Get positions and distance
        pos_diff = system.positions[current_idx] - system.positions[other_idx]
        distance = np.linalg.norm(pos_diff)
        
        print(f"\n=== Collision Detection ===")
        print(f"Distance: {distance}")
        
        current_chem = system.chemical_properties[current_idx]
        other_chem = system.chemical_properties[other_idx]
        
        print(f"Elements: {current_chem.element_data.id} and {other_chem.element_data.id}")
        print(f"Possible bonds: {current_chem.element_data.possible_bonds}")
        
        # Calculate collision energy
        relative_velocity = np.linalg.norm(system.velocities[current_idx] - system.velocities[other_idx])
        collision_energy = 0.5 * relative_velocity * relative_velocity
        
        print(f"Collision energy: {collision_energy}")
        print(f"Energy threshold: {ACTIVATION_ENERGY_THRESHOLD}")
        
        # First check if particles are already bonded
        already_bonded = any(
            bond.particle_id == other_idx 
            for bond in current_chem.bonds
        )
        
        if already_bonded:
            CollisionHandler._handle_bonded_particles(system, current_idx, other_idx)
            return True
            
        # Try chemical bonding if energy is appropriate (not too high or low)
        if ACTIVATION_ENERGY_THRESHOLD * 0.5 <= collision_energy <= ACTIVATION_ENERGY_THRESHOLD * 2.0:
            # Check if bonding is possible based on distance and element types
            combined_radius = (current_chem.element_data.radius + other_chem.element_data.radius) * 20.0  # Match display scaling
            if (distance <= combined_radius * 1.5 and  # Allow some flexibility in bonding distance
                current_chem.element_data.possible_bonds.get(other_chem.element_data.id)):
                
                # Try to form bond
                if CollisionHandler._try_form_bond(system, current_idx, other_idx, distance):
                    return True
        
        # If no bond formed, handle regular collision
        CollisionHandler._handle_regular_collision(system, current_idx, other_idx, pos_diff, distance)
        return False

    @staticmethod
    def _try_form_bond(system, idx1, idx2, distance) -> bool:
        """Try to form a bond between two particles"""
        # Check conditions including angle validity
        if not BondSystem.check_bond_formation_conditions(system, idx1, idx2, distance):
            return False
            
        if not CollisionHandler._check_bond_angle_validity(system, idx1, idx2):
            return False

        # Create bonds
        bonds = BondSystem.create_bond(system, idx1, idx2)
        if not bonds:
            return False

        bond1, bond2 = bonds
        
        # Add bonds to particles
        system.chemical_properties[idx1].bonds.append(bond1)
        system.chemical_properties[idx2].bonds.append(bond2)

        # Apply physical effects
        BondSystem.apply_bond_effects(system, idx1, idx2)

        return True

    @staticmethod
    def _try_molecular_reaction(system, idx1, idx2, collision_energy) -> bool:
        """Try to form complex molecular structures"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        # Check for aromatic ring formation
        if CollisionHandler._check_aromatic_ring_possible(system, idx1, idx2):
            return CollisionHandler._form_aromatic_ring(system, idx1, idx2)
            
        # Check for resonance structure formation
        if chem1.can_form_resonance(chem2):
            return CollisionHandler._form_resonance_structure(system, idx1, idx2)
            
        # Check for addition reactions (adding to double bonds)
        if CollisionHandler._check_addition_possible(system, idx1, idx2):
            return CollisionHandler._perform_addition_reaction(system, idx1, idx2)
            
        return False

    @staticmethod
    def _check_aromatic_ring_possible(system, idx1, idx2) -> bool:
        """Check if these atoms could form part of an aromatic ring"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        # Both must be sp2 hybridized
        if not (chem1.hybridization_state == 'sp2' and chem2.hybridization_state == 'sp2'):
            return False
            
        # Check if they're part of a potential ring
        if chem1.molecule and chem2.molecule and chem1.molecule == chem2.molecule:
            molecule = chem1.molecule
            
            # Count sp2 atoms in the molecule
            sp2_count = sum(1 for pid in molecule.atoms 
                          if system.chemical_properties[pid].hybridization_state == 'sp2')
            
            # Need 6 sp2 atoms for benzene-like aromaticity
            if sp2_count >= 5:
                # Check if this bond would complete the ring
                return CollisionHandler._would_complete_ring(system, idx1, idx2)
                
        return False

    @staticmethod
    def _form_aromatic_ring(system, idx1, idx2) -> bool:
        """Form an aromatic ring structure"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        # Form the aromatic bond
        success = chem1._form_aromatic_bond(chem2)
        if success:
            # Update all bonds in the ring to be aromatic
            molecule = chem1.molecule
            for atom_id in molecule.atoms:
                system.chemical_properties[atom_id].is_aromatic = True
                
            # Apply physical effects
            CollisionHandler._apply_ring_formation_effects(system, molecule)
            
        return success

    @staticmethod
    def _apply_ring_formation_effects(system, molecule):
        """Apply physical effects of ring formation"""
        # Calculate center of ring
        center = np.mean([system.positions[pid] for pid in molecule.atoms], axis=0)
        
        # Adjust positions to form regular polygon
        num_atoms = len(molecule.atoms)
        radius = PARTICLE_RADIUS * 2.5  # Slightly larger than particle diameter
        
        for i, pid in enumerate(molecule.atoms):
            angle = 2 * np.pi * i / num_atoms
            new_pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            
            # Smoothly move particle to new position
            current_pos = system.positions[pid]
            system.positions[pid] = current_pos * 0.2 + new_pos * 0.8
            
            # Add slight rotational velocity
            tangent = np.array([-np.sin(angle), np.cos(angle)])
            system.velocities[pid] = tangent * 0.5

    @staticmethod
    def _check_addition_possible(system, idx1, idx2) -> bool:
        """Check if addition reaction is possible (adding to double bond)"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        # Check if either molecule has a double bond
        has_double_bond = lambda chem: any(bond.order == 2.0 for bond in chem.bonds)
        
        return has_double_bond(chem1) or has_double_bond(chem2)

    @staticmethod
    def _perform_addition_reaction(system, idx1, idx2) -> bool:
        """Perform addition reaction across double bond"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        # Find the double bond
        if any(bond.order == 2.0 for bond in chem1.bonds):
            target_chem = chem1
            adding_chem = chem2
        else:
            target_chem = chem2
            adding_chem = chem1
            
        # Convert double bond to single bond
        double_bond = next(bond for bond in target_chem.bonds if bond.order == 2.0)
        double_bond.order = 1.0
        double_bond.shared_electrons = 2
        
        # Form new single bond with adding atom
        success = target_chem._form_covalent_bond(adding_chem)
        
        if success:
            # Update molecular geometry
            target_chem.molecule.molecular_geometry = target_chem._determine_molecular_geometry()
            
        return success

    @staticmethod
    def _form_resonance_structure(system, idx1, idx2) -> bool:
        """Form a resonance structure between atoms"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        success = chem1._form_resonance_bond(chem2)
        if success:
            # Apply physical effects of resonance
            CollisionHandler._apply_resonance_effects(system, chem1.molecule)
            
        return success

    @staticmethod
    def _apply_resonance_effects(system, molecule):
        """Apply physical effects of resonance structure formation"""
        # Add slight oscillation to represent electron delocalization
        for pid in molecule.atoms:
            # Add small random perpendicular velocity component
            vel = system.velocities[pid]
            perp = np.array([-vel[1], vel[0]])
            norm = np.linalg.norm(perp)
            if norm > 1e-6:
                perp /= norm
                system.velocities[pid] += perp * 0.3 * np.sin(system.time * 2)

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
        """Handle physics between bonded particles"""
        pos_diff = system.positions[idx1] - system.positions[idx2]
        distance = np.linalg.norm(pos_diff)
        
        # Use consistent distance multiplier
        target_distance = (
            system.chemical_properties[idx1].element_data.radius +
            system.chemical_properties[idx2].element_data.radius
        ) * 1.5  # Match chemical interaction distance
        
        if distance > 0:
            # Stronger correction to maintain bonds
            correction = (distance - target_distance) * 0.3
            direction = pos_diff / distance
            
            # Apply position correction
            system.positions[idx1] -= direction * correction
            system.positions[idx2] += direction * correction
            
            # Average velocities to keep bonded particles together
            avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
            system.velocities[idx1] = avg_velocity
            system.velocities[idx2] = avg_velocity

    @staticmethod
    def _check_bond_angle_validity(system, idx1, idx2) -> bool:
        """Check if bond formation would satisfy 2D geometry constraints"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        # Get hybridization states
        hybrid1 = chem1.hybridization_state
        hybrid2 = chem2.hybridization_state
        
        # 2D-specific angle constraints
        if hybrid1 == 'sp':
            target_angle = 180.0  # Linear in 2D
        elif hybrid1 == 'sp2':
            target_angle = 120.0  # Trigonal planar in 2D
        else:
            return True  # No specific angle constraint
        
        # Check existing bonds' angles
        for bond in chem1.bonds:
            angle = system.calculate_bond_angle(idx1, bond.particle_id, idx2)
            if angle is not None and abs(angle - target_angle) > BOND_ANGLE_TOLERANCE:
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
            
            if magnitudes == 0 or not np.isfinite(dot_product) or not np.isfinite(magnitudes):
                return None
                
            cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)  # Ensure valid range
            angle = np.arccos(cos_angle) * 180 / np.pi
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
                
                if magnitudes == 0 or not np.isfinite(dot_product) or not np.isfinite(magnitudes):
                    continue
                    
                cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)  # Ensure valid range
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
                
            return min(angles) if angles else None
            
        return None

    @staticmethod
    def _handle_chemical_interaction(system, idx1, idx2, collision_energy):
        """Enhanced chemical interaction handling"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        
        # Calculate distance
        pos_diff = system.positions[idx1] - system.positions[idx2]
        distance = np.sqrt(np.dot(pos_diff, pos_diff))
        
        print(f"\n=== Chemical Interaction ===")
        print(f"Distance: {distance}")
        
        # More lenient distance check based on particle radii
        combined_radius = (chem1.element_data.radius + chem2.element_data.radius) * 20  # Match display scaling
        print(f"Combined radius threshold: {combined_radius}")
        
        if distance > combined_radius * 1.5:  # Allow 50% more distance for bonding
            print("Too far for bonding")
            return False
        
        # Check if bonding is possible based on element properties
        bond_config = chem1.element_data.possible_bonds.get(chem2.element_data.id)
        print(f"Bond configuration: {bond_config}")
        
        if not bond_config:
            print("No valid bond configuration")
            return False
        
        # Check current bond counts
        print(f"Current bonds: {len(chem1.bonds)} and {len(chem2.bonds)}")
        print(f"Max bonds: {chem1.max_bonds} and {chem2.max_bonds}")
        
        # Try to form the bond if conditions are met
        if (len(chem1.bonds) < chem1.max_bonds and 
            len(chem2.bonds) < chem2.max_bonds):
            success = CollisionHandler._try_form_bond(system, idx1, idx2, distance)
            print(f"Bond formation result: {success}")
            if success:
                # Update particle velocities to reflect bonding
                CollisionHandler._apply_bonding_effects(system, idx1, idx2)
                return True
        else:
            print("Maximum bonds reached for one or both particles")
        
        return False

    @staticmethod
    def _apply_bonding_effects(system, idx1, idx2):
        """Apply physical effects of bond formation"""
        # Average the velocities with some energy release
        avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
        energy_release = 0.2  # Bond formation releases energy
        
        # Add some random perpendicular motion to simulate energy release
        perpendicular = np.array([-avg_velocity[1], avg_velocity[0]])
        perpendicular /= (np.linalg.norm(perpendicular) + 1e-6)
        
        system.velocities[idx1] = avg_velocity + perpendicular * energy_release
        system.velocities[idx2] = avg_velocity - perpendicular * energy_release

