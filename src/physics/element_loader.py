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
        
        try:
            print("\n=== Loading Element Data ===")
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for element_id, props in data.items():
                try:
                    print(f"\nLoading element: {element_id}")
                    elements[element_id] = ElementProperties(**props)
                except Exception as e:
                    print(f"Error loading element {element_id}: {e}")
                    continue
                    
            if not elements:
                raise ValueError("No valid elements loaded")
                
            return elements
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Elements data file not found at {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in elements file: {file_path}") 