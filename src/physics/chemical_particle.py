from dataclasses import dataclass
from typing import List, Set, Optional
import numpy as np
from core.element_data import ELEMENT_DATA, ElementProperties

@dataclass
class Bond:
    particle_id: int
    bond_type: str
    strength: float
    angle: Optional[float] = None  # Store bond angle for proper geometry

class ChemicalParticle:
    def __init__(self, element_id: str, particle_id: int):
        self.element_data: ElementProperties = ELEMENT_DATA[element_id]
        self.particle_id = particle_id
        self.bonds: List[Bond] = []
        self.current_charge = 0
        self.temperature = 298.15  # Room temperature in Kelvin
        self.energy = 0.0
        self.electron_count = self.element_data.valence_electrons
        self.hybridization = None
        self.bond_angles = {}  # Store actual angles between bonds
        
    def can_bond_with(self, other: 'ChemicalParticle') -> bool:
        if other.element_data.id in self.element_data.possible_bonds:
            bond_info = self.element_data.possible_bonds[other.element_data.id]
            
            # Check valence electron availability
            if self.electron_count < 2 or other.electron_count < 2:
                return False
                
            # Check maximum bonds based on hybridization
            max_bonds = bond_info['max_bonds']
            if len(self.bonds) >= max_bonds:
                return False
                
            # Check bond angle constraints
            if len(self.bonds) > 0 and 'angle' in bond_info:
                return self._check_bond_angle_validity(other, bond_info['angle'])
                
            return True
        return False
    
    def _check_bond_angle_validity(self, other: 'ChemicalParticle', target_angle: float) -> bool:
        """Check if new bond would satisfy angle constraints"""
        if len(self.bonds) == 0:
            return True
            
        # This is a placeholder - actual implementation would need position data
        # from the particle system to calculate real angles
        return True  # For now, always allow
        
    def try_form_bond(self, other: 'ChemicalParticle', distance: float) -> bool:
        if not self.can_bond_with(other):
            return False
            
        # Check distance threshold for bonding
        bond_distance = 2.0 * (self.element_data.radius + other.element_data.radius)
        if distance > bond_distance:
            return False
            
        # Check activation energy threshold
        activation_energy = (self.element_data.activation_energy + 
                           other.element_data.activation_energy) * 0.5
        if self.temperature < activation_energy:
            return False
            
        bond_info = self.element_data.possible_bonds[other.element_data.id]
        new_bond = Bond(
            particle_id=other.particle_id,
            bond_type=bond_info['type'],
            strength=bond_info['strength'],
            angle=bond_info.get('angle')
        )
        
        # Update electron counts
        electrons_shared = 2  # Single bond
        self.electron_count -= electrons_shared // 2
        other.electron_count -= electrons_shared // 2
        
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
