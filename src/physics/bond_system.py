from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from core.constants import BOND_ENERGY_SCALE, ACTIVATION_ENERGY_THRESHOLD
from physics.chemical_particle import Bond

@dataclass
class BondConfiguration:
    type: str
    strength: float
    shared_electrons: int
    hybridization: Optional[str] = None
    resonance: bool = False
    order: float = 1.0

class BondSystem:
    @staticmethod
    def determine_bond_type(en_diff: float) -> str:
        """Determine bond type based on electronegativity difference"""
        if en_diff > 1.7:
            return 'ionic'
        elif en_diff > 0.4:
            return 'polar_covalent'
        return 'covalent'

    @staticmethod
    def calculate_bond_strength(bond_type: str, base_strength: float) -> float:
        """Calculate final bond strength based on type and base value"""
        strength_multipliers = {
            'ionic': 0.8,
            'polar_covalent': 1.2,
            'covalent': 1.0,
            'metallic': 0.6
        }
        return base_strength * strength_multipliers.get(bond_type, 1.0) * BOND_ENERGY_SCALE

    @staticmethod
    def check_bond_formation_conditions(system, idx1: int, idx2: int, distance: float) -> bool:
        """Check if bond formation is possible between two particles"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]

        # Check existing bonds
        if any(bond.particle_id == idx2 for bond in chem1.bonds):
            return False

        # Check max bonds
        if len(chem1.bonds) >= chem1.max_bonds or len(chem2.bonds) >= chem2.max_bonds:
            return False

        # Get bond configuration
        bond_config = chem1.element_data.possible_bonds.get(chem2.element_data.id)
        if not bond_config:
            return False

        # Check distance with more lenient threshold
        combined_radius = (chem1.element_data.radius + chem2.element_data.radius) * 2.0
        ideal_bond_distance = combined_radius * 1.2  # Slightly larger than touching
        
        # Allow bonding within a range around ideal distance
        min_distance = ideal_bond_distance * 0.7
        max_distance = ideal_bond_distance * 1.5
        
        if not (min_distance <= distance <= max_distance):
            return False

        return True

    @staticmethod
    def create_bond(system, idx1: int, idx2: int) -> Optional[tuple[Bond, Bond]]:
        """Create appropriate bonds between two particles"""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]

        # Get base configuration
        config = chem1.element_data.possible_bonds.get(chem2.element_data.id)
        if not config:
            return None

        # Determine bond type
        en_diff = abs(chem1.element_data.electronegativity - chem2.element_data.electronegativity)
        bond_type = BondSystem.determine_bond_type(en_diff)
        
        # Calculate final strength
        strength = BondSystem.calculate_bond_strength(bond_type, config['strength'])

        # Create bonds
        bond1 = Bond(
            particle_id=idx2,
            bond_type=bond_type,
            strength=strength,
            shared_electrons=config.get('shared_electrons', 2),
            hybridization=chem1.hybridization_state
        )

        bond2 = Bond(
            particle_id=idx1,
            bond_type=bond_type,
            strength=strength,
            shared_electrons=config.get('shared_electrons', 2),
            hybridization=chem2.hybridization_state
        )

        return bond1, bond2

    @staticmethod
    def apply_bond_effects(system, idx1: int, idx2: int):
        """Apply physical effects of bond formation"""
        # Average velocities with energy release
        avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
        energy_release = 0.2

        # Add perpendicular motion for energy release
        perpendicular = np.array([-avg_velocity[1], avg_velocity[0]])
        perpendicular /= (np.linalg.norm(perpendicular) + 1e-6)

        system.velocities[idx1] = avg_velocity + perpendicular * energy_release
        system.velocities[idx2] = avg_velocity - perpendicular * energy_release 