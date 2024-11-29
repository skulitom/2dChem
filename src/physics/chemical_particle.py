from dataclasses import dataclass
from typing import List, Set, Optional, Dict
import numpy as np
from core.element_data import ELEMENT_DATA, ElementProperties

@dataclass
class Bond:
    particle_id: int
    bond_type: str  # 'covalent', 'ionic', or 'metallic'
    strength: float
    shared_electrons: int  # Number of electrons shared (2 for single, 4 for double, 6 for triple)
    angle: Optional[float] = None

class ChemicalParticle:
    def __init__(self, element_id: str, particle_id: int):
        self.element_data: ElementProperties = ELEMENT_DATA[element_id]
        self.particle_id = particle_id
        self.bonds: List[Bond] = []
        
        # 2D-specific electron configuration
        self.valence_electrons = self.element_data.valence_electrons
        self.current_charge = 0
        self.electron_configuration = self._init_2d_electron_config()
        
        # Physical state properties
        self.temperature = 298.15  # Room temperature in Kelvin
        self.phase_state = self.element_data.phase_state
        self.energy = 0.0
        
        # Hybridization and geometry
        self.hybridization = None
        self.preferred_geometry = self._determine_preferred_geometry()

    def _init_2d_electron_config(self) -> Dict[str, int]:
        """Initialize 2D electron configuration following sextet rule"""
        config = {}
        remaining_electrons = self.valence_electrons
        
        # Fill orbitals according to 2D rules
        orbital_order = ['1s', '2s', '2p', '3s', '3p', '3d']
        max_electrons = {'s': 2, 'p': 4, 'd': 4, 'f': 4}  # 2D orbital capacities
        
        for orbital in orbital_order:
            if remaining_electrons <= 0:
                break
            orbital_type = orbital[-1]
            capacity = max_electrons[orbital_type]
            electrons_added = min(remaining_electrons, capacity)
            config[orbital] = electrons_added
            remaining_electrons -= electrons_added
            
        return config

    def _determine_preferred_geometry(self) -> Dict[str, float]:
        """Determine preferred bond angles based on valence electrons"""
        valence = self.valence_electrons
        if valence <= 2:
            return {'linear': 180.0}
        elif valence <= 4:
            return {'trigonal': 120.0}
        else:
            return {'triangular': 120.0}  # 2D equivalent of tetrahedral

    def can_form_ionic_bond(self, other: 'ChemicalParticle') -> bool:
        """Check if ionic bond formation is possible based on electronegativity"""
        en_diff = abs(self.element_data.electronegativity - other.element_data.electronegativity)
        return en_diff > 1.7  # Threshold for ionic bonding

    def can_bond_with(self, other: 'ChemicalParticle', distance: float) -> bool:
        # Check sextet rule (max 6 valence electrons in 2D)
        if len(self.bonds) >= 3 or len(other.bonds) >= 3:  # Max 3 bonds in 2D
            return False

        # Check if ionic or covalent bonding is possible
        if self.can_form_ionic_bond(other):
            return True

        # Check covalent bonding possibility
        if other.element_data.id in self.element_data.possible_bonds:
            bond_info = self.element_data.possible_bonds[other.element_data.id]
            
            # Check distance threshold
            bond_distance = (self.element_data.radius + other.element_data.radius) * 1.2
            if distance > bond_distance:
                return False
            
            # Check geometry constraints
            if len(self.bonds) > 0:
                return self._check_2d_geometry_constraints(other)
            
            return True
        return False

    def _check_2d_geometry_constraints(self, other: 'ChemicalParticle') -> bool:
        """Check if new bond would satisfy 2D geometry constraints"""
        if len(self.bonds) == 0:
            return True
            
        # This is a placeholder - actual implementation would need position data
        # from the particle system to calculate real angles
        return True  # For now, always allow
        
    def try_form_bond(self, other: 'ChemicalParticle', distance: float) -> bool:
        if not self.can_bond_with(other):
            return False
            
        # Check distance threshold for bonding
        bond_distance = (self.element_data.radius + other.element_data.radius) * 1.2
        if distance > bond_distance:
            return False
            
        # Lower activation energy threshold for testing
        activation_energy_threshold = 0.1  # Make it easier to form bonds initially
        if self.temperature < activation_energy_threshold:
            return False
            
        bond_info = self.element_data.possible_bonds[other.element_data.id]
        new_bond = Bond(
            particle_id=other.particle_id,
            bond_type=bond_info['type'],
            strength=bond_info['strength'],
            shared_electrons=bond_info['shared_electrons'],
            angle=bond_info.get('angle')
        )
        
        # Update electron counts
        electrons_shared = 2  # Single bond
        self.electron_count -= electrons_shared // 2
        other.electron_count -= electrons_shared // 2
        
        # Add some attraction between bonded particles
        self.bonds.append(new_bond)
        return True
    
    def break_all_bonds(self):
        """Break all bonds when temperature exceeds boiling point or other conditions"""
        self.bonds.clear()        
    def break_bond(self, other_particle_id: int) -> bool:
        """Break a specific bond with another particle"""
        for i, bond in enumerate(self.bonds):
            if bond.particle_id == other_particle_id:
                # Return electrons when breaking bond
                electrons_shared = 2  # Single bond
                self.electron_count += electrons_shared // 2
                # Note: other particle needs to be updated separately
                
                self.bonds.pop(i)
                return True
        return False
    
    def get_total_bond_energy(self) -> float:
        """Calculate total bond energy"""
        return sum(bond.strength for bond in self.bonds)
    
    def update_temperature(self, ambient_temp: float, delta_time: float):
        """Update particle temperature based on ambient temperature and bonding"""
        # Temperature changes more slowly when bonded
        cooling_factor = 1.0 / (1.0 + len(self.bonds) * 0.5)
        
        # Consider phase transitions
        if self.temperature >= self.element_data.boiling_point:
            self.break_all_bonds()  # Molecules break apart when boiling
        elif self.temperature <= self.element_data.melting_point:
            cooling_factor *= 0.5  # Slower temperature changes in solid state
        
        # Simple temperature equilibrium with ambient
        temp_diff = ambient_temp - self.temperature
        self.temperature += temp_diff * cooling_factor * delta_time
        
        # Add heat from bond energies and consider heat of formation
        bond_heat = (self.get_total_bond_energy() * 0.1 + 
                    self.element_data.heat_of_formation * 0.05)
        self.temperature += bond_heat * delta_time
