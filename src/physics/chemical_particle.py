from dataclasses import dataclass
from typing import List, Set, Optional, Dict
import numpy as np
from core.element_data import ELEMENT_DATA, ElementProperties
from core.constants import DEBUG_MODE

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
    def __init__(self, element_id: str, particle_id: int):
        self.element_data: ElementProperties = ELEMENT_DATA[element_id]
        self.particle_id = particle_id
        self.bonds: List[Bond] = []
        
        # Basic properties
        self.valence_electrons = self.element_data.valence_electrons
        self.current_charge = 0
        self.temperature = 298.15  # Room temperature in Kelvin
        self.phase_state = self.element_data.phase_state
        self.energy = 0.0
        
        # Initialize electron configuration first
        self.electron_shells = self._init_2d_electron_shells()
        self.hybridization_state = self._determine_hybridization()
        
        # Then calculate derived properties
        self.max_bonds = self._calculate_max_bonds()
        self.lone_pairs = self._calculate_lone_pairs()
        self.bonding_electrons = 0
        self.electron_domains = self._calculate_electron_domains()
        
        # Molecular properties
        self.molecule: Optional[Molecule] = None
        self.is_aromatic = False
        self.resonance_contributor = False
        self.formal_charge = 0
        self.electronegativity = self.element_data.electronegativity
        
        # Enhanced 2D electron configuration
        self.electron_configuration = self._init_2d_electron_config()
        
        # Hybridization and geometry
        self.hybridization = None
        self.preferred_geometry = self._determine_preferred_geometry()
        
        # Track ionic charge separately from valence electrons
        self.ionic_charge = 0
        
        # Enhanced electron tracking
        self.electron_domains = self._calculate_electron_domains()
        
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

    def _init_2d_electron_shells(self) -> Dict[int, int]:
        """Initialize electron shells following 2D quantum mechanics"""
        shells = {}
        electrons = self.element_data.atomic_number
        
        # Fill shells according to 2D rules (2, 6, 10, 14...)
        shell_capacities = [2, 6, 10, 14]  # Maximum electrons per shell in 2D
        
        shell_num = 1
        while electrons > 0:
            capacity = shell_capacities[min(shell_num - 1, len(shell_capacities) - 1)]
            electrons_in_shell = min(electrons, capacity)
            shells[shell_num] = electrons_in_shell
            electrons -= electrons_in_shell
            shell_num += 1
            
        return shells

    def _determine_hybridization(self) -> Optional[str]:
        """Determine orbital hybridization based on valence electrons and bonding"""
        if self.valence_electrons <= 2:
            return 'sp'  # Linear geometry (180°)
        elif self.valence_electrons <= 6:
            return 'sp2'  # Trigonal planar (120°)
        return None

    def can_form_ionic_bond(self, other: 'ChemicalParticle') -> bool:
        """Enhanced ionic bond check based on electronegativity difference"""
        en_diff = abs(self.electronegativity - other.electronegativity)
        
        # Check if ionic bond formation would lead to sextet rule satisfaction
        if en_diff > 1.7:  # Threshold for ionic character
            # Calculate resulting electron configurations
            if self.electronegativity < other.electronegativity:
                donor, acceptor = self, other
            else:
                donor, acceptor = other, self
                
            # Check if electron transfer would satisfy sextet rule
            donor_remaining = donor.valence_electrons - 1
            acceptor_new = acceptor.valence_electrons + 1
            
            return (donor_remaining in [0, 2, 6] or  # Stable configurations in 2D
                    acceptor_new in [2, 6])  # Sextet rule
                    
        return False

    def can_bond_with(self, other: 'ChemicalParticle', distance: float) -> bool:
        """Check if bonding is possible between two particles"""
        # Check if elements can bond with each other
        if other.element_data.id not in self.element_data.possible_bonds:
            return False
        
        if self.element_data.id not in other.element_data.possible_bonds:
            return False

        # Check max bonds
        if len(self.bonds) >= 3 or len(other.bonds) >= 3:
            return False

        # Calculate exact bonding distance based on radii (scaled to match display)
        combined_radius = (self.element_data.radius + other.element_data.radius) * 20  # Match display scaling
        
        # In 2D, particles should touch exactly at their edges for bonding
        # Allow only a very small tolerance for numerical stability
        min_dist = combined_radius * 0.95  # 5% tolerance below
        max_dist = combined_radius * 1.05  # 5% tolerance above
        
        if not (min_dist <= distance <= max_dist):
            return False

        return True
        
    def _check_2d_geometry_constraints(self, other: 'ChemicalParticle') -> bool:
        """Check if new bond would satisfy 2D geometry constraints"""
        if len(self.bonds) == 0:
            return True
            
        # This is a placeholder - actual implementation would need position data
        # from the particle system to calculate real angles
        return True  # For now, always allow
        
    def try_form_bond(self, other: 'ChemicalParticle', distance: float) -> bool:
        """Enhanced bonding logic with molecular formation"""
        if len(self.bonds) >= self.max_bonds:
            return False
            
        # First check for resonance structure formation
        if self.can_form_resonance(other):
            return self._form_resonance_bond(other)
            
        # Check for aromatic ring formation
        if self._can_form_aromatic_ring(other):
            return self._form_aromatic_bond(other)
            
        # Try ionic bonding
        if self.can_form_ionic_bond(other):
            return self._form_ionic_bond(other)
            
        # Try covalent bonding with molecular orbital consideration
        if self.can_form_covalent_bond(other):
            return self._form_covalent_bond(other)
            
        return False
        
    def _form_resonance_bond(self, other: 'ChemicalParticle') -> bool:
        """Form a resonance bond between atoms"""
        new_bond = Bond(
            particle_id=other.particle_id,
            bond_type='covalent',
            strength=1.5,  # Intermediate strength for resonance
            shared_electrons=3,  # 1.5 bonds worth of electrons
            hybridization='sp2',
            resonance=True,
            order=1.5
        )
        
        self.bonds.append(new_bond)
        self.resonance_contributor = True
        other.resonance_contributor = True
        
        # Update molecular structure
        if self.molecule:
            self.molecule.resonance_structures = True
            if other.molecule != self.molecule:
                other.molecule = self.molecule
                self.molecule.atoms.append(other.particle_id)
                
        return True
        
    def _can_form_aromatic_ring(self, other: 'ChemicalParticle') -> bool:
        """Check if atoms can form part of an aromatic ring"""
        if not (self.hybridization_state == 'sp2' and other.hybridization_state == 'sp2'):
            return False
            
        # Check if this would complete a ring of sp2 hybridized atoms
        if self.molecule and other.molecule and self.molecule == other.molecule:
            molecule = self.molecule
            sp2_atoms = sum(1 for pid in molecule.atoms 
                          if molecule.chemical_properties[pid].hybridization_state == 'sp2')
            return sp2_atoms >= 5  # Potential for aromaticity
            
        return False
        
    def _form_aromatic_bond(self, other: 'ChemicalParticle') -> bool:
        """Form an aromatic bond between atoms"""
        new_bond = Bond(
            particle_id=other.particle_id,
            bond_type='covalent',
            strength=1.5,
            shared_electrons=3,
            hybridization='sp2',
            resonance=True,
            order=1.5
        )
        
        self.bonds.append(new_bond)
        self.is_aromatic = True
        other.is_aromatic = True
        
        # Update molecular structure
        if self.molecule:
            self.molecule.aromaticity = True
            if other.molecule != self.molecule:
                other.molecule = self.molecule
                self.molecule.atoms.append(other.particle_id)
                
        return True
        
    def _form_covalent_bond(self, other: 'ChemicalParticle') -> bool:
        """Form a covalent bond with consideration of molecular orbitals"""
        # Determine bond order based on available electrons and hybridization
        if (self.hybridization_state == 'sp2' and other.hybridization_state == 'sp2' and
            self.valence_electrons >= 4 and other.valence_electrons >= 4):
            shared_electrons = 4
            bond_order = 2.0
        else:
            shared_electrons = 2
            bond_order = 1.0
            
        # Create bonds for both particles
        bond_self = Bond(
            particle_id=other.particle_id,
            bond_type='covalent',
            strength=1.0 * bond_order,
            shared_electrons=shared_electrons,
            hybridization=self.hybridization_state,
            order=bond_order
        )
        
        bond_other = Bond(
            particle_id=self.particle_id,
            bond_type='covalent',
            strength=1.0 * bond_order,
            shared_electrons=shared_electrons,
            hybridization=other.hybridization_state,
            order=bond_order
        )
        
        self.bonds.append(bond_self)
        other.bonds.append(bond_other)
        self.bonding_electrons += shared_electrons // 2
        other.bonding_electrons += shared_electrons // 2
        
        # Create or update molecular structure
        if not self.molecule:
            self.molecule = Molecule(
                atoms=[self.particle_id, other.particle_id],
                bonds=[bond_self, bond_other],
                molecular_geometry=self._determine_molecular_geometry(),
                total_charge=self.ionic_charge + other.ionic_charge
            )
            other.molecule = self.molecule
        else:
            self.molecule.atoms.append(other.particle_id)
            self.molecule.bonds.extend([bond_self, bond_other])
            other.molecule = self.molecule
            
        return True
        
    def _determine_molecular_geometry(self) -> str:
        """Determine molecular geometry based on electron domains"""
        domains = self._calculate_electron_domains()
        if domains <= 2:
            return 'linear'
        elif domains == 3:
            return 'trigonal_planar'
        else:
            return 'triangular'  # 2D equivalent of tetrahedral

    def _calculate_lone_pairs(self) -> int:
        """Calculate number of lone pairs based on valence electrons"""
        bonded_electrons = sum(bond.shared_electrons // 2 for bond in self.bonds)
        return (self.valence_electrons - bonded_electrons) // 2
        
    def _calculate_electron_domains(self) -> int:
        """Calculate electron domains (bonding + lone pairs)"""
        return len(self.bonds) + self.lone_pairs
        
    def can_form_resonance(self, other: 'ChemicalParticle') -> bool:
        """Check if resonance structure formation is possible"""
        if not (self.hybridization_state == 'sp2' and other.hybridization_state == 'sp2'):
            return False
            
        # Check for alternating single/double bond pattern
        if self.molecule and other.molecule and self.molecule == other.molecule:
            single_bonds = sum(1 for bond in self.bonds if bond.order == 1.0)
            double_bonds = sum(1 for bond in self.bonds if bond.order == 2.0)
            return single_bonds > 0 and double_bonds > 0
            
        return False
        
    def _form_ionic_bond(self, other: 'ChemicalParticle') -> bool:
        """Form an ionic bond between atoms"""
        if self.electronegativity < other.electronegativity:
            donor, acceptor = self, other
        else:
            donor, acceptor = other, self
            
        # Update charges and electrons
        donor.ionic_charge += 1
        acceptor.ionic_charge -= 1
        donor.valence_electrons -= 1
        acceptor.valence_electrons += 1
        
        # Create ionic bond
        new_bond = Bond(
            particle_id=acceptor.particle_id,
            bond_type='ionic',
            strength=1.2,  # Ionic bonds are typically stronger
            shared_electrons=0,  # No shared electrons in ionic bonds
            hybridization=None
        )
        donor.bonds.append(new_bond)
        
        # Update molecular structure
        if not donor.molecule:
            donor.molecule = Molecule(
                atoms=[donor.particle_id, acceptor.particle_id],
                bonds=[new_bond],
                molecular_geometry='ionic',
                total_charge=0  # Net charge is zero for ionic compounds
            )
            acceptor.molecule = donor.molecule
        else:
            donor.molecule.atoms.append(acceptor.particle_id)
            donor.molecule.bonds.append(new_bond)
            acceptor.molecule = donor.molecule
            
        return True
        
    def break_all_bonds(self):
        """Break all bonds when temperature exceeds boiling point or other conditions"""
        self.bonds.clear()        
    def break_bond(self, other_particle_id: int) -> bool:
        """Break a specific bond with another particle"""
        for i, bond in enumerate(self.bonds):
            if bond.particle_id == other_particle_id:
                # Return electrons when breaking bond
                self.bonding_electrons -= bond.shared_electrons // 2
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

    def _calculate_max_bonds(self) -> int:
        """Calculate maximum possible bonds based on hybridization and valence electrons"""
        if self.hybridization_state == 'sp':
            return 2  # Linear geometry
        elif self.hybridization_state == 'sp2':
            return 3  # Trigonal planar
        else:
            # Default based on valence electrons in 2D
            return min(self.valence_electrons, 3)  # Maximum of 3 bonds in 2D

    def can_form_covalent_bond(self, other: 'ChemicalParticle') -> bool:
        """Check if covalent bond formation is possible based on 2D sextet rule"""
        # Check current number of valence electrons
        total_valence = self.valence_electrons + other.valence_electrons
        
        # Check if bonding would exceed sextet (6 electrons) for either atom
        self_after_bond = self.valence_electrons + 1
        other_after_bond = other.valence_electrons + 1
        
        # In 2D, atoms are stable with 2 or 6 electrons (sextet rule)
        stable_counts = [2, 6]
        
        # Check if bond formation would lead to stable configurations
        would_be_stable = (
            self_after_bond in stable_counts or
            other_after_bond in stable_counts
        )
        
        return (
            len(self.bonds) < self.max_bonds and
            len(other.bonds) < other.max_bonds and
            would_be_stable
        )
