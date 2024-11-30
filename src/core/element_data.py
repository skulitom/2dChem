from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class ElementProperties:
    id: str
    name: str
    atomic_number: int
    mass: float
    radius: float
    color: Tuple[int, int, int]
    
    # 2D Quantum Properties
    electron_configuration: Dict[str, int]  # e.g., {'1s': 2, '2s': 2, '2p': 4}
    valence_electrons: int  # Max 6 in 2D
    principal_quantum_numbers: List[int]  # n values
    angular_momentum_numbers: List[int]  # l values
    magnetic_quantum_numbers: List[float]  # mₛ values
    
    # Chemical Properties
    electronegativity: float
    oxidation_states: List[int]
    phase_state: str
    density: float
    
    # Thermal Properties
    melting_point: float
    boiling_point: float
    sublimation_point: Optional[float]
    activation_energy: float
    heat_of_formation: float
    
    # Bonding Properties
    possible_bonds: Dict[str, Dict]
    hybridization_states: List[str]  # ['s', 'sp1', 'sp2']
    bond_angles: Dict[str, float]  # e.g., {'sp2': 120.0}
    
    # Special Behaviors
    special_behaviors: List[str]  # ['magnetic', 'radioactive', 'catalyst']
    catalyst_affinities: Dict[str, float]

# Define base elements according to our 2D chemistry rules
ELEMENT_DATA = {
    'H': ElementProperties(
        id='H',
        name='Hydrogen',
        atomic_number=1,
        mass=1.0,
        radius=0.32,
        color=(255, 255, 255),
        electron_configuration={'1s': 1},
        valence_electrons=1,
        principal_quantum_numbers=[1],
        angular_momentum_numbers=[0],
        magnetic_quantum_numbers=[0.0],
        electronegativity=2.2,
        oxidation_states=[1],
        phase_state='gas',
        density=0.09,
        melting_point=14.01,
        boiling_point=20.28,
        sublimation_point=None,
        activation_energy=0.1,
        heat_of_formation=0.0,
        possible_bonds={
            'O': {'type': 'covalent', 'strength': 2, 'max_bonds': 1},
            'N': {'type': 'covalent', 'strength': 1, 'max_bonds': 1},
            'C': {'type': 'covalent', 'strength': 1, 'max_bonds': 1}
        },
        hybridization_states=['s'],
        bond_angles={'s': 180.0},
        special_behaviors=['flammable'],
        catalyst_affinities={'Pt': 0.9, 'Pd': 0.8, 'Ni': 0.7}
    ),
    'O': ElementProperties(
        id='O',
        name='Oxygen',
        atomic_number=8,
        mass=16.0,
        radius=0.66,
        color=(255, 0, 0),
        electron_configuration={'1s': 2, '2s': 2, '2p': 2},
        valence_electrons=4,
        principal_quantum_numbers=[1, 2],
        angular_momentum_numbers=[0, 1],
        magnetic_quantum_numbers=[0.0, 1.0],
        electronegativity=3.5,
        oxidation_states=[2],
        phase_state='gas',
        density=1.43,
        melting_point=54.36,
        boiling_point=90.20,
        sublimation_point=None,
        activation_energy=0.2,
        heat_of_formation=-0.1,
        possible_bonds={
            'H': {'type': 'covalent', 'strength': 2, 'max_bonds': 2},
            'C': {'type': 'covalent', 'strength': 2, 'max_bonds': 2},
            'N': {'type': 'covalent', 'strength': 2, 'max_bonds': 1}
        },
        hybridization_states=['sp2'],
        bond_angles={'sp2': 120.0},
        special_behaviors=['oxidizer'],
        catalyst_affinities={'Pt': 0.7, 'Pd': 0.6}
    ),
    'N': ElementProperties(
        id='N',
        name='Nitrogen',
        atomic_number=7,
        mass=14.0,
        radius=0.71,
        color=(0, 0, 255),
        electron_configuration={'1s': 2, '2s': 2, '2p': 3},
        valence_electrons=5,
        principal_quantum_numbers=[1, 2],
        angular_momentum_numbers=[0, 1],
        magnetic_quantum_numbers=[0.0, 0.0],
        electronegativity=3.0,
        oxidation_states=[3],
        phase_state='gas',
        density=1.25,
        melting_point=63.15,
        boiling_point=77.36,
        sublimation_point=None,
        activation_energy=0.0,
        heat_of_formation=0.0,
        possible_bonds={
            'H': {'type': 'covalent', 'strength': 1, 'max_bonds': 3},
            'O': {'type': 'covalent', 'strength': 2, 'max_bonds': 2},
            'C': {'type': 'covalent', 'strength': 1, 'max_bonds': 3}
        },
        hybridization_states=['sp1', 'sp2'],
        bond_angles={'sp2': 120.0, 'sp1': 180.0},
        special_behaviors=[],
        catalyst_affinities={'Fe': 0.8, 'Ru': 0.7}
    ),
    'C': ElementProperties(
        id='C',
        name='Carbon',
        atomic_number=6,
        mass=12.0,
        radius=0.76,
        color=(128, 128, 128),
        electron_configuration={'1s': 2, '2s': 2, '2p': 2},
        valence_electrons=4,
        principal_quantum_numbers=[1, 2],
        angular_momentum_numbers=[0, 1],
        magnetic_quantum_numbers=[0.0, 0.0],
        electronegativity=2.5,
        oxidation_states=[4],
        phase_state='solid',
        density=2.27,
        melting_point=3800,
        boiling_point=4300,
        sublimation_point=None,
        activation_energy=0.0,
        heat_of_formation=0.0,
        possible_bonds={
            'H': {'type': 'covalent', 'strength': 1, 'max_bonds': 3, 'angle': 120},
            'O': {'type': 'covalent', 'strength': 2, 'max_bonds': 2, 'angle': 120},
            'N': {'type': 'covalent', 'strength': 1, 'max_bonds': 3, 'angle': 120},
            'C': {'type': 'covalent', 'strength': 1, 'max_bonds': 3, 'angle': 120}
        },
        hybridization_states=['sp1', 'sp2'],
        bond_angles={'sp2': 120.0},
        special_behaviors=['forms_chains'],
        catalyst_affinities={'Pt': 0.8, 'Pd': 0.7}
    ),
    'He': ElementProperties(
        id='He',
        name='Helium',
        atomic_number=2,
        mass=4.0,
        radius=0.28,
        color=(255, 255, 128),
        electron_configuration={'1s': 2},
        valence_electrons=2,
        principal_quantum_numbers=[1],
        angular_momentum_numbers=[0],
        magnetic_quantum_numbers=[0.0],
        electronegativity=0.0,
        oxidation_states=[0],
        phase_state='gas',
        density=0.17,
        melting_point=0.95,
        boiling_point=4.22,
        sublimation_point=None,
        activation_energy=0.0,
        heat_of_formation=0.0,
        possible_bonds={},
        hybridization_states=[],
        bond_angles={},
        special_behaviors=['noble_gas'],
        catalyst_affinities={}
    ),
    'Li': ElementProperties(
        id='Li',
        name='Lithium',
        atomic_number=3,
        mass=6.94,
        radius=1.52,
        color=(204, 128, 255),
        electron_configuration={'1s': 2, '2s': 1},
        valence_electrons=1,
        principal_quantum_numbers=[1, 2],
        angular_momentum_numbers=[0, 1],
        magnetic_quantum_numbers=[0.0, 0.0],
        electronegativity=0.98,
        oxidation_states=[1],
        phase_state='solid',
        density=0.53,
        melting_point=453.69,
        boiling_point=1615,
        sublimation_point=None,
        activation_energy=0.05,
        heat_of_formation=-0.2,
        possible_bonds={
            'O': {'type': 'ionic', 'strength': 3, 'max_bonds': 1},
            'F': {'type': 'ionic', 'strength': 3, 'max_bonds': 1},
            'Cl': {'type': 'ionic', 'strength': 2, 'max_bonds': 1}
        },
        hybridization_states=['s'],
        bond_angles={},
        special_behaviors=['metallic'],
        catalyst_affinities={'Pt': 0.3, 'Pd': 0.2}
    )
} 