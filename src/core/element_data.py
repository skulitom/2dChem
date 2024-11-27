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
    valence_electrons: int
    electronegativity: float
    phase_state: str
    density: float
    melting_point: float
    boiling_point: float
    decomposition_temp: Optional[float]
    special_behaviors: List[str]
    possible_bonds: Dict[str, Dict]
    electron_configuration: Dict[str, int]
    hybridization_states: List[str]
    bond_angles: Dict[str, float]
    activation_energy: float
    heat_of_formation: float
    catalyst_affinities: Dict[str, float]

# Define base elements according to our 2D chemistry rules
ELEMENT_DATA = {
    'H': ElementProperties(
        id='H',
        name='Hydrogen',
        atomic_number=1,
        mass=1.0,
        radius=0.5,
        color=(255, 255, 255),
        valence_electrons=1,
        electronegativity=2.2,
        phase_state='gas',
        density=0.09,
        melting_point=14.01,
        boiling_point=20.28,
        decomposition_temp=None,
        special_behaviors=['flammable'],
        possible_bonds={
            'O': {'type': 'covalent', 'strength': 2, 'max_bonds': 1},
            'N': {'type': 'covalent', 'strength': 1, 'max_bonds': 1},
            'C': {'type': 'covalent', 'strength': 1, 'max_bonds': 1}
        },
        electron_configuration={'1s': 1},
        hybridization_states=['s'],
        bond_angles={'s': 180.0},
        activation_energy=432.0,
        heat_of_formation=0.0,
        catalyst_affinities={'Pt': 0.9, 'Pd': 0.8, 'Ni': 0.7}
    ),
    'O': ElementProperties(
        id='O',
        name='Oxygen',
        atomic_number=8,
        mass=16.0,
        radius=0.6,
        color=(255, 0, 0),
        valence_electrons=6,
        electronegativity=3.5,
        phase_state='gas',
        density=1.43,
        melting_point=54.36,
        boiling_point=90.20,
        decomposition_temp=None,
        special_behaviors=['oxidizer'],
        possible_bonds={
            'H': {'type': 'covalent', 'strength': 2, 'max_bonds': 2},
            'C': {'type': 'covalent', 'strength': 2, 'max_bonds': 2},
            'N': {'type': 'covalent', 'strength': 2, 'max_bonds': 1}
        },
        electron_configuration={'1s': 2, '2s': 2, '2p': 4},
        hybridization_states=['sp2'],
        bond_angles={'sp2': 120.0},
        activation_energy=498.0,
        heat_of_formation=0.0,
        catalyst_affinities={'Pt': 0.7, 'Pd': 0.6}
    ),
    'N': ElementProperties(
        id='N',
        name='Nitrogen',
        atomic_number=7,
        mass=14.0,
        radius=0.55,
        color=(0, 0, 255),
        valence_electrons=5,
        electronegativity=3.0,
        phase_state='gas',
        density=1.25,
        melting_point=63.15,
        boiling_point=77.36,
        decomposition_temp=None,
        special_behaviors=[],
        possible_bonds={
            'H': {'type': 'covalent', 'strength': 1, 'max_bonds': 3},
            'O': {'type': 'covalent', 'strength': 2, 'max_bonds': 2},
            'C': {'type': 'covalent', 'strength': 1, 'max_bonds': 3}
        },
        electron_configuration={'1s': 2, '2s': 2, '2p': 3},
        hybridization_states=['sp1', 'sp2'],
        bond_angles={'sp2': 120.0, 'sp1': 180.0},
        activation_energy=945.3,
        heat_of_formation=0.0,
        catalyst_affinities={'Fe': 0.8, 'Ru': 0.7}
    ),
    'C': ElementProperties(
        id='C',
        name='Carbon',
        atomic_number=6,
        mass=12.0,
        radius=0.58,
        color=(128, 128, 128),
        valence_electrons=4,
        electronegativity=2.5,
        phase_state='solid',
        density=2.27,
        melting_point=3800,
        boiling_point=4300,
        decomposition_temp=None,
        special_behaviors=['forms_chains'],
        possible_bonds={
            'H': {'type': 'covalent', 'strength': 1, 'max_bonds': 3, 'angle': 120},
            'O': {'type': 'covalent', 'strength': 2, 'max_bonds': 2, 'angle': 120},
            'N': {'type': 'covalent', 'strength': 1, 'max_bonds': 3, 'angle': 120},
            'C': {'type': 'covalent', 'strength': 1, 'max_bonds': 3, 'angle': 120}
        },
        electron_configuration={'1s': 2, '2s': 2, '2p': 2},
        hybridization_states=['sp1', 'sp2'],
        bond_angles={'sp2': 120.0},
        activation_energy=348.0,
        heat_of_formation=0.0,
        catalyst_affinities={'Pt': 0.8, 'Pd': 0.7}
    ),
    'He': ElementProperties(
        id='He',
        name='Helium',
        atomic_number=2,
        mass=4.0,
        radius=0.45,
        color=(255, 255, 128),
        valence_electrons=2,
        electronegativity=0.0,
        phase_state='gas',
        density=0.17,
        melting_point=0.95,
        boiling_point=4.22,
        decomposition_temp=None,
        special_behaviors=['noble_gas'],
        possible_bonds={},  # Noble gas, forms no bonds
        electron_configuration={'1s': 2},
        hybridization_states=[],  # Noble gases don't hybridize
        bond_angles={},  # No bonds, no angles
        activation_energy=0.0,  # Inert, doesn't react
        heat_of_formation=0.0,
        catalyst_affinities={}
    ),
    'Li': ElementProperties(
        id='Li',
        name='Lithium',
        atomic_number=3,
        mass=6.94,
        radius=0.68,
        color=(204, 128, 255),
        valence_electrons=1,
        electronegativity=0.98,
        phase_state='solid',
        density=0.53,
        melting_point=453.69,
        boiling_point=1615,
        decomposition_temp=None,
        special_behaviors=['metallic'],
        possible_bonds={
            'O': {'type': 'ionic', 'strength': 3, 'max_bonds': 1},
            'F': {'type': 'ionic', 'strength': 3, 'max_bonds': 1},
            'Cl': {'type': 'ionic', 'strength': 2, 'max_bonds': 1}
        },
        electron_configuration={'1s': 2, '2s': 1},
        hybridization_states=['s'],  # Metallic bonding
        bond_angles={},  # Ionic bonds don't have fixed angles
        activation_energy=159.3,  # kJ/mol, typical for Li reactions
        heat_of_formation=0.0,
        catalyst_affinities={'Pt': 0.3, 'Pd': 0.2}
    )
} 