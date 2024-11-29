import json
from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class ElementProperties:
    id: str
    name: str
    atomic_number: int
    mass: float
    radius: float
    electron_configuration: Dict[str, int]
    valence_electrons: int
    electronegativity: float
    phase_state: str
    melting_point: float
    boiling_point: float
    possible_bonds: Dict[str, Dict[str, Any]]

class ElementLoader:
    @staticmethod
    def load_elements() -> Dict[str, ElementProperties]:
        elements = {}
        file_path = os.path.join(os.path.dirname(__file__), '../data/elements.json')
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for element_id, props in data.items():
            elements[element_id] = ElementProperties(
                id=element_id,
                **props
            )
            
        return elements 