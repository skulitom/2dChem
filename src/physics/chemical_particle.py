from dataclasses import dataclass
from typing import List, Set, Optional
import numpy as np
from core.element_data import ELEMENT_DATA, ElementProperties

@dataclass
class Bond:
    particle_id: int
    bond_type: str
    strength: float

class ChemicalParticle:
    def __init__(self, element_id: str, particle_id: int):
        self.element_data: ElementProperties = ELEMENT_DATA[element_id]
        self.particle_id = particle_id
        self.bonds: List[Bond] = []
        self.current_charge = 0
        self.temperature = 298.15  # Room temperature in Kelvin
        self.energy = 0.0
        
    def can_bond_with(self, other: 'ChemicalParticle') -> bool:
        if other.element_data.id in self.element_data.possible_bonds:
            bond_info = self.element_data.possible_bonds[other.element_data.id]
            return len(self.bonds) < bond_info['max_bonds']
        return False
    
    def try_form_bond(self, other: 'ChemicalParticle', distance: float) -> bool:
        if not self.can_bond_with(other):
            return False
            
        # Check distance threshold for bonding
        if distance > 2.0 * (self.element_data.radius + other.element_data.radius):
            return False
            
        bond_info = self.element_data.possible_bonds[other.element_data.id]
        new_bond = Bond(
            particle_id=other.particle_id,
            bond_type=bond_info['type'],
            strength=bond_info['strength']
        )
        self.bonds.append(new_bond)
        return True
    
    def break_all_bonds(self):
        """Break all bonds when temperature exceeds boiling point or other conditions"""
        self.bonds.clear()
        
    def break_bond(self, other_particle_id: int) -> bool:
        """Break a specific bond with another particle"""
        for i, bond in enumerate(self.bonds):
            if bond.particle_id == other_particle_id:
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
        
        # Simple temperature equilibrium with ambient
        temp_diff = ambient_temp - self.temperature
        self.temperature += temp_diff * cooling_factor * delta_time
        
        # Add heat from bond energies
        bond_heat = self.get_total_bond_energy() * 0.1  # Scale factor for gameplay
        self.temperature += bond_heat * delta_time