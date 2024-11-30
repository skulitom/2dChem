import pygame
from pygame import Color
import numpy as np

# Window Settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Grid Settings
CELL_SIZE = 8
GRID_WIDTH = WINDOW_WIDTH // CELL_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // CELL_SIZE

# Particle Settings
PARTICLE_RADIUS = 4
COLLISION_DAMPING = 0.5
GRAVITY = np.float32(9.8)
PARTICLES_PER_FRAME = 1
PARTICLE_SPREAD = 8
COLLISION_RESPONSE = 0.7
MIN_PARTICLES = 1
MAX_PARTICLES = 20

# Colors
BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)

# Physics Settings
FIXED_TIMESTEP = 1/60
MAX_STEPS_PER_FRAME = 1
VELOCITY_SCALE = 5.0

# Chemical simulation constants
BOND_DISTANCE_THRESHOLD = 1.5
TEMPERATURE_TRANSFER_RATE = 0.1
REACTION_PROBABILITY = 0.5
AMBIENT_TEMPERATURE = 298.15

# Update element colors
ELEMENT_COLORS = {
    'H': (200, 200, 220),    # Softer hydrogen
    'O': (235, 80, 80),      # Rich red
    'N': (90, 130, 245),     # Vibrant blue
    'C': (75, 75, 90),       # Refined carbon
    'He': (255, 255, 128), # Light yellow for Helium
    'Li': (204, 128, 255), # Light purple for Lithium
    'Na': (255, 128, 0),   # Orange for Sodium
    'Cl': (0, 255, 0),     # Green for Chlorine
    'F': (255, 255, 0),    # Yellow for Fluorine
    'P': (255, 128, 128),  # Pink for Phosphorus
    'S': (255, 255, 0),    # Yellow for Sulfur
    'K': (163, 0, 163),    # Purple for Potassium
    'Ca': (128, 128, 0),   # Olive for Calcium
    'Fe': (128, 0, 0),     # Dark red for Iron
    'Cu': (255, 128, 0),   # Orange for Copper
    'Zn': (128, 128, 128), # Gray for Zinc
    'Ag': (192, 192, 192), # Silver for Silver
    'Au': (255, 215, 0),   # Gold for Gold
    'Hg': (190, 190, 190), # Light gray for Mercury
    'Pb': (128, 128, 128)  # Gray for Lead
}

# Add new physics constants
BOND_ENERGY_SCALE = 0.8
TEMPERATURE_VISUALIZATION_SCALE = 1000.0
ACTIVATION_ENERGY_THRESHOLD = 0.05

# Reaction constants
COLLISION_ANGLE_TOLERANCE = 45.0
CATALYST_EFFICIENCY_FACTOR = 2.5
ELECTRON_TRANSFER_RATE = 0.15

# Visual feedback
REACTION_PARTICLES = {
    'BOND_FORM': (0, 255, 0),  # Green sparkles for bond formation
    'BOND_BREAK': (255, 0, 0),  # Red sparkles for bond breaking
    'CATALYST': (0, 255, 255),  # Cyan sparkles for catalysis
}

# Update UI colors for better contrast and cleaner look
UI_COLORS = {
    'BACKGROUND': (10, 10, 15),       # Darker background
    'PANEL': (22, 22, 30),           # Slightly darker panel
    'TEXT': (220, 220, 235),         # Softer white
    'HIGHLIGHT': (100, 140, 255),    # Brighter blue highlight
    'BORDER': (40, 40, 55),          # Slightly darker border
    'TAB_ACTIVE': (75, 75, 95),      # Slightly darker active tab
    'TAB_HOVER': (32, 32, 45),       # Refined hover state
}

# Simulation Frame Offsets
SIMULATION_X_OFFSET = 200  # Sidebar width
SIMULATION_Y_OFFSET = 40   # Tab height
SIMULATION_WIDTH = WINDOW_WIDTH - SIMULATION_X_OFFSET
SIMULATION_HEIGHT = WINDOW_HEIGHT - SIMULATION_Y_OFFSET

# Add or update these constants
MAX_VELOCITY = 30.0
FLOOR_BUFFER = 10  # Small buffer from bottom of screen
MIN_BOUNCE_VELOCITY = 0.1

# 2D Physics Constants
ELECTROMAGNETIC_CONSTANT = 1.0  # Adjusted for 2D
BOND_ANGLE_TOLERANCE = 5.0  # Degrees
MAX_BONDS_2D = 3  # Maximum bonds per atom in 2D
SEXTET_RULE = 6  # Maximum valence electrons in 2D

# Orbital Energy Levels (2D-specific)
ORBITAL_ENERGIES = {
    '1s': -13.6,
    '2s': -3.4,
    '2p': -1.7,
    '3s': -0.85,
    '3p': -0.43,
    '3d': -0.25,
}

# 2D Hybridization Angles
HYBRIDIZATION_ANGLES = {
    'sp1': 180.0,  # Linear
    'sp2': 120.0,  # Trigonal planar
}

# Physics Constants
BOND_FORCE_SCALE = 2.0
MIN_REACTION_VELOCITY = 0.05
MAX_REACTION_VELOCITY = 5.0

# Chemical Constants
ACTIVATION_ENERGY_SCALE = 0.1
TEMPERATURE_SCALE = 0.01
BOND_ENERGY_SCALE = 0.5
ACTIVATION_ENERGY_THRESHOLD = 0.1

# Debug settings
DEBUG_MODE = True
DEBUG_SHOW_BONDS = True
DEBUG_SHOW_BOND_COUNT = True
DEBUG_COLLISION_RADIUS = PARTICLE_RADIUS * 4.0

print("\n=== Window Configuration ===")
print(f"Window dimensions: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
print(f"Simulation area: {SIMULATION_WIDTH}x{SIMULATION_HEIGHT}")
print(f"Particle radius: {PARTICLE_RADIUS}")
print(f"Offsets: X={SIMULATION_X_OFFSET}, Y={SIMULATION_Y_OFFSET}")