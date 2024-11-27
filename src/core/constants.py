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
COLLISION_DAMPING = 0.75
GRAVITY = np.float32(9.8)
PARTICLES_PER_FRAME = 1
PARTICLE_SPREAD = 10
COLLISION_RESPONSE = 0.5
MIN_PARTICLES = 1
MAX_PARTICLES = 20

# Colors
BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)

# Physics Settings
FIXED_TIMESTEP = 1/10  # Physics updates at 60Hz
MAX_STEPS_PER_FRAME = 3
VELOCITY_SCALE = 5.0  # New constant for initial velocity scaling

# Chemical simulation constants
BOND_DISTANCE_THRESHOLD = 2.0
TEMPERATURE_TRANSFER_RATE = 0.1
REACTION_PROBABILITY = 0.5
AMBIENT_TEMPERATURE = 298.15  # Room temperature in Kelvin

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
COLLISION_ANGLE_TOLERANCE = 30.0  # degrees
CATALYST_EFFICIENCY_FACTOR = 2.0
ELECTRON_TRANSFER_RATE = 0.1

# Visual feedback
REACTION_PARTICLES = {
    'BOND_FORM': (0, 255, 0),  # Green sparkles for bond formation
    'BOND_BREAK': (255, 0, 0),  # Red sparkles for bond breaking
    'CATALYST': (0, 255, 255),  # Cyan sparkles for catalysis
}