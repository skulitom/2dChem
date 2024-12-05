from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from core.constants import BOND_ENERGY_SCALE, ACTIVATION_ENERGY_THRESHOLD
from physics.chemical_particle import Bond

@dataclass
class BondConfiguration:
    type: str = 'covalent'
    strength: float = 1.0
    shared_electrons: int = 2
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
    def _load_bond_configuration(chem1, chem2) -> Optional[BondConfiguration]:
        """
        Helper method to load and unify bond configuration from two particles.
        Checks if both sides have a defined bonding possibility to ensure symmetry.
        """
        config_1 = chem1.element_data.possible_bonds.get(chem2.element_data.id)
        config_2 = chem2.element_data.possible_bonds.get(chem1.element_data.id)

        # If either side does not define a bond, we consider it impossible
        if not config_1 or not config_2:
            return None

        # Merge configurations for consistency. For simplicity, we’ll take the
        # average strength if both define it. Other parameters are taken from chem1's config.
        # You can customize merging logic as needed.
        strength = (config_1.get('strength', 1.0) + config_2.get('strength', 1.0)) / 2.0

        return BondConfiguration(
            strength=strength,
            shared_electrons=config_1.get('shared_electrons', 2),
            hybridization=config_1.get('hybridization', None),
            resonance=config_1.get('resonance', False),
            order=config_1.get('order', 1.0)
        )

    @staticmethod
    def check_bond_formation_conditions(system, idx1: int, idx2: int, distance: float) -> bool:
        """Check if bond formation is possible between two particles considering symmetry."""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]

        # Check if already bonded
        if any(bond.particle_id == idx2 for bond in chem1.bonds):
            return False

        # Check max bond count
        if len(chem1.bonds) >= chem1.max_bonds or len(chem2.bonds) >= chem2.max_bonds:
            return False

        # Load symmetrical bond configuration
        bond_config = BondSystem._load_bond_configuration(chem1, chem2)
        if not bond_config:
            return False

        # Distance checks
        combined_radius = (chem1.element_data.radius + chem2.element_data.radius) * 2.0
        ideal_bond_distance = combined_radius * 1.2  # Slightly larger than touching

        min_distance = ideal_bond_distance * 0.7
        max_distance = ideal_bond_distance * 1.5

        if not (min_distance <= distance <= max_distance):
            return False

        return True

    @staticmethod
    def create_bond(system, idx1: int, idx2: int) -> Optional[Tuple[Bond, Bond]]:
        """Create bonds between two particles using a symmetrical bond configuration."""
        chem1 = system.chemical_properties[idx1]
        chem2 = system.chemical_properties[idx2]

        # Load symmetrical bond configuration
        bond_config = BondSystem._load_bond_configuration(chem1, chem2)
        if not bond_config:
            return None

        # Determine bond type
        en_diff = abs(chem1.element_data.electronegativity - chem2.element_data.electronegativity)
        bond_type = BondSystem.determine_bond_type(en_diff)

        # Calculate final strength and ensure it's above some threshold if desired
        strength = BondSystem.calculate_bond_strength(bond_type, bond_config.strength)
        if strength < ACTIVATION_ENERGY_THRESHOLD:
            return None

        bond1 = Bond(
            particle_id=idx2,
            bond_type=bond_type,
            strength=strength,
            shared_electrons=bond_config.shared_electrons,
            hybridization=chem1.hybridization_state
        )

        bond2 = Bond(
            particle_id=idx1,
            bond_type=bond_type,
            strength=strength,
            shared_electrons=bond_config.shared_electrons,
            hybridization=chem2.hybridization_state
        )

        return bond1, bond2

    @staticmethod
    def apply_bond_effects(system, idx1: int, idx2: int):
        """Apply physical effects of bond formation such as energy release and velocity adjustments."""
        avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
        energy_release = 0.2

        # Create a perpendicular vector to avg_velocity
        perpendicular = np.array([-avg_velocity[1], avg_velocity[0]])
        norm = np.linalg.norm(perpendicular)
        if norm == 0:
            # If no motion, give a small random kick
            perpendicular = np.array([1.0, 0.0])
        else:
            perpendicular /= (norm + 1e-6)

        system.velocities[idx1] = avg_velocity + perpendicular * energy_release
        system.velocities[idx2] = avg_velocity - perpendicular * energy_release
