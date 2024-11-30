from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ElementProperties:
    id: str
    name: str
    atomic_number: int
    mass: float
    radius: float
    color: tuple
    electron_configuration: Dict[str, int]
    valence_electrons: int
    principal_quantum_numbers: List[int]
    angular_momentum_numbers: List[int]
    magnetic_quantum_numbers: List[float]
    electronegativity: float
    oxidation_states: List[int]
    phase_state: str
    density: float
    melting_point: float
    boiling_point: float
    sublimation_point: Optional[float]
    activation_energy: float
    heat_of_formation: float
    possible_bonds: Dict[str, Dict[str, float]]
    hybridization_states: List[str]
    bond_angles: Dict[str, float]
    special_behaviors: List[str]
    catalyst_affinities: Dict[str, float]

# Define base elements
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
        electron_configuration={'1s': 2, '2s': 2, '2p': 4},
        valence_electrons=6,
        principal_quantum_numbers=[1, 2],
        angular_momentum_numbers=[0, 1],
        magnetic_quantum_numbers=[0.0, 1.0],
        electronegativity=3.44,
        oxidation_states=[-2],
        phase_state='gas',
        density=1.429,
        melting_point=54.36,
        boiling_point=90.20,
        sublimation_point=None,
        activation_energy=0.2,
        heat_of_formation=0.0,
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
        electronegativity=3.04,
        oxidation_states=[-3],
        phase_state='gas',
        density=1.251,
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
        electronegativity=2.55,
        oxidation_states=[4, -4],
        phase_state='solid',
        density=2.267,
        melting_point=3800,
        boiling_point=4300,
        sublimation_point=None,
        activation_energy=0.0,
        heat_of_formation=0.0,
        possible_bonds={
            'H': {'type': 'covalent', 'strength': 1, 'max_bonds': 4},
            'O': {'type': 'covalent', 'strength': 2, 'max_bonds': 2},
            'N': {'type': 'covalent', 'strength': 1, 'max_bonds': 3},
            'C': {'type': 'covalent', 'strength': 1, 'max_bonds': 4}
        },
        hybridization_states=['sp1', 'sp2', 'sp3'],
        bond_angles={'sp3': 109.5, 'sp2': 120.0, 'sp1': 180.0},
        special_behaviors=['forms_chains'],
        catalyst_affinities={'Pt': 0.8, 'Pd': 0.7}
    )
} 