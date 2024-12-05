import numpy as np
from core.constants import (
    COLLISION_RESPONSE,
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
        pos_diff = system.positions[current_idx] - system.positions[other_idx]
        distance = np.linalg.norm(pos_diff)

        current_chem = system.chemical_properties[current_idx]
        other_chem = system.chemical_properties[other_idx]

        # Calculate collision energy
        relative_velocity = np.linalg.norm(system.velocities[current_idx] - system.velocities[other_idx])
        collision_energy = 0.5 * relative_velocity * relative_velocity

        # Check if particles are already bonded
        already_bonded = any(bond.particle_id == other_idx for bond in current_chem.bonds)
        if already_bonded:
            CollisionHandler._handle_bonded_particles(system, current_idx, other_idx)
            return True

        # Attempt bond formation if energy is within reasonable range
        if ACTIVATION_ENERGY_THRESHOLD * 0.5 <= collision_energy <= ACTIVATION_ENERGY_THRESHOLD * 2.0:
            combined_radius = (current_chem.element_data.radius + other_chem.element_data.radius) * 20.0
            if (distance <= combined_radius * 1.5 and
                current_chem.element_data.possible_bonds.get(other_chem.element_data.id)):

                if CollisionHandler._try_form_bond(system, current_idx, other_idx, distance):
                    return True

        # If no bond formed, handle regular collision
        CollisionHandler._handle_regular_collision(system, current_idx, other_idx, pos_diff, distance)
        return False

    @staticmethod
    def _try_form_bond(system, idx1, idx2, distance) -> bool:
        """Try to form a bond between two particles"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]

        # Check bond formation conditions first
        if not BondSystem.check_bond_formation_conditions(system, idx1, idx2, distance):
            combined_radius = (chem1.element_data.radius + chem2.element_data.radius) * 20.0
            ideal_bond_distance = combined_radius * 1.2
            min_distance = ideal_bond_distance * 0.7
            max_distance = ideal_bond_distance * 1.5

            if distance < min_distance or distance > max_distance:
                return False

        # Check bond angle validity
        if not CollisionHandler._check_bond_angle_validity(system, idx1, idx2):
            return False

        # Attempt actual bond creation
        bonds = BondSystem.create_bond(system, idx1, idx2)
        if not bonds:
            return False

        bond1, bond2 = bonds
        system.chemical_properties[idx1].bonds.append(bond1)
        system.chemical_properties[idx2].bonds.append(bond2)

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

        # Check for addition reactions
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

        # Check if they're in the same molecule
        if chem1.molecule and chem2.molecule and chem1.molecule == chem2.molecule:
            molecule = chem1.molecule

            # Count sp2 atoms in the molecule
            sp2_count = sum(1 for pid in molecule.atoms
                            if system.chemical_properties[pid].hybridization_state == 'sp2')

            # Need at least 5 sp2 atoms before completing the 6th for aromaticity
            if sp2_count >= 5:
                return CollisionHandler._would_complete_ring(system, idx1, idx2)

        return False

    @staticmethod
    def _would_complete_ring(system, idx1, idx2) -> bool:
        """Check if adding a bond between idx1 and idx2 would complete a ring.
           This is a placeholder; implement proper ring detection as needed."""
        return False

    @staticmethod
    def _form_aromatic_ring(system, idx1, idx2) -> bool:
        """Form an aromatic ring structure"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]

        success = chem1._form_aromatic_bond(chem2)
        if success and chem1.molecule:
            molecule = chem1.molecule
            for atom_id in molecule.atoms:
                system.chemical_properties[atom_id].is_aromatic = True
            CollisionHandler._apply_ring_formation_effects(system, molecule)
        return success

    @staticmethod
    def _apply_ring_formation_effects(system, molecule):
        """Apply physical effects of ring formation"""
        center = np.mean([system.positions[pid] for pid in molecule.atoms], axis=0)
        
        # Compute an average radius based on the average particle radius in the molecule
        avg_radius = np.mean([system.chemical_properties[pid].element_data.radius for pid in molecule.atoms])
        radius = avg_radius * 2.5
        
        num_atoms = len(molecule.atoms)
        for i, pid in enumerate(molecule.atoms):
            angle = 2 * np.pi * i / num_atoms
            new_pos = center + radius * np.array([np.cos(angle), np.sin(angle)])

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

        has_double_bond = lambda chem: any(bond.order == 2.0 for bond in chem.bonds)
        return has_double_bond(chem1) or has_double_bond(chem2)

    @staticmethod
    def _perform_addition_reaction(system, idx1, idx2) -> bool:
        """Perform addition reaction across double bond"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]

        if any(bond.order == 2.0 for bond in chem1.bonds):
            target_chem = chem1
            adding_chem = chem2
        else:
            target_chem = chem2
            adding_chem = chem1

        double_bond = next(bond for bond in target_chem.bonds if bond.order == 2.0)
        double_bond.order = 1.0
        double_bond.shared_electrons = 2

        success = target_chem._form_covalent_bond(adding_chem)
        if success and target_chem.molecule:
            target_chem.molecule.molecular_geometry = target_chem._determine_molecular_geometry()
        return success

    @staticmethod
    def _form_resonance_structure(system, idx1, idx2) -> bool:
        """Form a resonance structure between atoms"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]

        success = chem1._form_resonance_bond(chem2)
        if success and chem1.molecule:
            CollisionHandler._apply_resonance_effects(system, chem1.molecule)
        return success

    @staticmethod
    def _apply_resonance_effects(system, molecule):
        """Apply physical effects of resonance structure formation"""
        for pid in molecule.atoms:
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

        if collision_energy > ACTIVATION_ENERGY_THRESHOLD:
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

        if collision_energy > ACTIVATION_ENERGY_THRESHOLD * 0.5:
            angle = CollisionHandler._calculate_bond_angle(system, idx1, idx2)
            if angle is not None:
                expected_angle = chem1.preferred_geometry.get('trigonal', 120.0)
                if abs(angle - expected_angle) > 30.0:
                    return

            shared_electrons = min(2, chem1.valence_electrons, chem2.valence_electrons)
            if shared_electrons > 0:
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
                    angle=None
                )

                chem1.bonds.append(bond1)
                chem2.bonds.append(bond2)

                avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
                system.velocities[idx1] = avg_velocity * 0.9
                system.velocities[idx2] = avg_velocity * 0.9

    @staticmethod
    @profile_function(threshold_ms=0.5)
    def _handle_regular_collision(system, idx1, idx2, pos_diff, distance):
        if distance <= 0:
            return

        direction = pos_diff / distance
        relative_velocity = system.velocities[idx1] - system.velocities[idx2]
        approach_speed = np.dot(relative_velocity, direction)

        # Only apply a collision response if they're actually moving towards each other
        if approach_speed < 0:
            damping = COLLISION_DAMPING
            impulse = direction * approach_speed * COLLISION_RESPONSE * damping

            mass1 = system.chemical_properties[idx1].element_data.mass
            mass2 = system.chemical_properties[idx2].element_data.mass
            total_mass = mass1 + mass2

            system.velocities[idx1] -= impulse * (mass2 / total_mass)
            system.velocities[idx2] += impulse * (mass1 / total_mass)

        overlap = ((system.chemical_properties[idx1].element_data.radius +
                    system.chemical_properties[idx2].element_data.radius) - distance)
        if overlap > 0:
            separation = direction * overlap * 0.3
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
        ) * 1.5

        if distance > 0:
            correction = (distance - target_distance) * 0.3
            direction = pos_diff / distance

            system.positions[idx1] -= direction * correction
            system.positions[idx2] += direction * correction

            avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
            system.velocities[idx1] = avg_velocity
            system.velocities[idx2] = avg_velocity

    @staticmethod
    def _check_bond_angle_validity(system, idx1, idx2) -> bool:
        chem1 = system.chemical_properties[idx1]
        hybrid1 = chem1.hybridization_state

        if hybrid1 == 'sp':
            target_angle = 180.0
        elif hybrid1 == 'sp2':
            target_angle = 120.0
        else:
            # If it's another hybridization, assume no strict angle check
            return True

        # Check angles with existing bonds
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
        particle1 = system.chemical_properties[idx1]

        if len(particle1.bonds) == 0:
            return None

        pos1 = system.positions[idx1]
        pos2 = system.positions[idx2]

        # One existing bond
        if len(particle1.bonds) == 1:
            existing_bond = particle1.bonds[0]
            pos_existing = system.positions[existing_bond.particle_id]

            vec1 = pos_existing - pos1
            vec2 = pos2 - pos1

            dot_product = np.dot(vec1, vec2)
            magnitudes = np.linalg.norm(vec1) * np.linalg.norm(vec2)

            if magnitudes == 0 or not np.isfinite(dot_product) or not np.isfinite(magnitudes):
                return None

            cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            return angle

        # Two existing bonds
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

                cos_angle = np.clip(dot_product / magnitudes, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)

            return min(angles) if angles else None

        return None

    @staticmethod
    def _handle_chemical_interaction(system, idx1, idx2, collision_energy):
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]
        pos_diff = system.positions[idx1] - system.positions[idx2]
        distance = np.linalg.norm(pos_diff)

        bond_config = chem1.element_data.possible_bonds.get(chem2.element_data.id)
        if not bond_config:
            return False

        if (len(chem1.bonds) < chem1.max_bonds and
            len(chem2.bonds) < chem2.max_bonds):
            success = CollisionHandler._try_form_bond(system, idx1, idx2, distance)
            if success:
                CollisionHandler._apply_bonding_effects(system, idx1, idx2)
                return True
        return False

    @staticmethod
    def _apply_bonding_effects(system, idx1, idx2):
        avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
        energy_release = 0.2

        perpendicular = np.array([-avg_velocity[1], avg_velocity[0]])
        norm = np.linalg.norm(perpendicular)
        if norm > 1e-6:
            perpendicular /= norm

        system.velocities[idx1] = avg_velocity + perpendicular * energy_release
        system.velocities[idx2] = avg_velocity - perpendicular * energy_release
