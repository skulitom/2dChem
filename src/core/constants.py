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
PARTICLE_RADIUS = 3
COLLISION_DAMPING = 0.5
GRAVITY = np.float32(4.0)
PARTICLES_PER_FRAME = 1
PARTICLE_SPREAD = 8
COLLISION_RESPONSE = 0.3
MIN_PARTICLES = 1
MAX_PARTICLES = 20

# Colors
BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)

# Physics Settings
FIXED_TIMESTEP = 1/10
MAX_STEPS_PER_FRAME = 1
VELOCITY_SCALE = 5.0

# Chemical simulation constants
BOND_DISTANCE_THRESHOLD = 1.5
TEMPERATURE_TRANSFER_RATE = 0.1
REACTION_PROBABILITY = 0.5
AMBIENT_TEMPERATURE = 298.15

# Update element colors
ELEMENT_COLORS = {
    'H': Color(255, 255, 255),  # White
    'O': Color(255, 0, 0),      # Red
    'N': Color(0, 0, 255),      # Blue
    'C': Color(128, 128, 128),  # Gray
    'He': Color(255, 255, 128), # Light yellow
    'Li': Color(204, 128, 255), # Light purple
    # Add more elements as needed
}

# Add new physics constants
BOND_ENERGY_SCALE = 0.5
TEMPERATURE_VISUALIZATION_SCALE = 1000.0
ACTIVATION_ENERGY_THRESHOLD = 100.0

# Reaction constants
COLLISION_ANGLE_TOLERANCE = 30.0
CATALYST_EFFICIENCY_FACTOR = 2.0
ELECTRON_TRANSFER_RATE = 0.1

# Visual feedback
REACTION_PARTICLES = {
    'BOND_FORM': (0, 255, 0),  # Green sparkles for bond formation
    'BOND_BREAK': (255, 0, 0),  # Red sparkles for bond breaking
    'CATALYST': (0, 255, 255),  # Cyan sparkles for catalysis
}

# Add new UI color constants
UI_COLORS = {
    'BACKGROUND': (15, 15, 20),  # Dark blue-black
    'PANEL': (30, 30, 40),       # Slightly lighter panel background
    'TEXT': (220, 220, 220),     # Soft white
    'HIGHLIGHT': (65, 105, 225),  # Royal blue
    'BORDER': (60, 60, 80),      # Subtle border color
    'TAB_ACTIVE': (80, 80, 100), # Active tab background
    'TAB_HOVER': (50, 50, 65),   # Tab hover state
}

# Simulation Frame Offsets
SIMULATION_X_OFFSET = 200  # Sidebar width
SIMULATION_Y_OFFSET = 40   # Tab height
SIMULATION_WIDTH = WINDOW_WIDTH - SIMULATION_X_OFFSET
SIMULATION_HEIGHT = WINDOW_HEIGHT - SIMULATION_Y_OFFSET

# Add or update these constants
MAX_VELOCITY = 20.0
FLOOR_BUFFER = 4
MIN_BOUNCE_VELOCITY = 0.5

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