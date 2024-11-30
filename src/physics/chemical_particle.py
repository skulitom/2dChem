from dataclasses import dataclass
from typing import List, Set, Optional, Dict
import numpy as np
from core.element_data import ELEMENT_DATA, ElementProperties
from core.constants import DEBUG_MODE
from .element_loader import ElementLoader

@dataclass
class Bond:
    particle_id: int
    bond_type: str  # 'covalent', 'ionic', 'metallic'
    strength: float
    shared_electrons: int
    angle: Optional[float] = None
    hybridization: Optional[str] = None  # 'sp', 'sp2'
    resonance: bool = False  # For delocalized electrons
    order: float = 1.0  # Can be 1.0, 1.5 (resonance), or 2.0 (double bond)

@dataclass
class Molecule:
    """Represents a molecular structure in 2D"""
    atoms: List[int]  # particle IDs
    bonds: List[Bond]
    molecular_geometry: str  # 'linear', 'bent', 'trigonal_planar', 'triangular'
    total_charge: int
    resonance_structures: bool = False
    aromaticity: bool = False

class ChemicalParticle:
    def __init__(self, element_type: str, particle_id: int):
        self.element_data = ElementLoader.get_element(element_type)
        self.particle_id = particle_id
        self.bonds = []
        self.max_bonds = self._calculate_max_bonds()
        self.molecule = None  # Reference to containing molecule if part of one
        self.is_aromatic = False
        
    def _calculate_max_bonds(self) -> int:
        """Calculate maximum number of bonds based on valence electrons"""
        # For simple elements, use valence electron count
        if self.element_data.id in ['H', 'F', 'Cl', 'Br', 'I']:
            return 1
        elif self.element_data.id in ['O']:
            return 2
        elif self.element_data.id in ['N']:
            return 3
        elif self.element_data.id in ['C']:
            return 4
        return 1  # Default to 1 for unknown elements
        
    def break_all_bonds(self):
        """Break all bonds this particle has with others"""
        if not hasattr(self, 'bonds'):
            return
        
        # Store bonds in a new list to avoid modifying while iterating
        try:
            bonds_to_break = self.bonds.copy() if self.bonds else []
        except Exception:
            bonds_to_break = []
        
        # Clear bonds first to prevent circular references
        self.bonds = []
        
        # Now break bonds with other particles
        for bond in bonds_to_break:
            try:
                if not hasattr(self, 'particle_system') or not bond or not hasattr(bond, 'particle_id'):
                    continue
                
                other_particle = self.particle_system.chemical_properties.get(bond.particle_id)
                if other_particle and hasattr(other_particle, 'bonds'):
                    try:
                        # Make a copy to avoid modifying while iterating
                        other_bonds = other_particle.bonds.copy()
                        other_particle.bonds = [
                            b for b in other_bonds 
                            if hasattr(b, 'particle_id') and b.particle_id != self.particle_id
                        ]
                    except Exception as e:
                        print(f"Warning: Error updating other particle bonds: {e}")
            except Exception as e:
                print(f"Warning: Error breaking bond: {e}")
                continue
        
        # Clean up molecule reference
        try:
            if hasattr(self, 'molecule') and self.molecule:
                if hasattr(self.molecule, 'atoms'):
                    try:
                        # Make a copy to avoid modifying while iterating
                        atoms = self.molecule.atoms.copy()
                        self.molecule.atoms = [
                            pid for pid in atoms 
                            if pid != self.particle_id
                        ]
                    except Exception as e:
                        print(f"Warning: Error updating molecule atoms: {e}")
                self.molecule = None
        except Exception as e:
            print(f"Warning: Error cleaning up molecule: {e}")
            self.molecule = None
        
        # Reset state
        self.is_aromatic = False
        self.bonds = []  # Ensure bonds are cleared even if there were errors
