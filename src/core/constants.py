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
PARTICLE_SPREAD = 15
COLLISION_RESPONSE = 0.5
MIN_PARTICLES = 1
MAX_PARTICLES = 20

# Colors
BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)

# Physics Settings
FIXED_TIMESTEP = 1/10  # Increased physics update rate
MAX_STEPS_PER_FRAME = 2  # Reduced to prevent physics bottleneck
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

# Simulation Frame Settings
SIMULATION_FRAME_X_OFFSET = 200  # Sidebar width
SIMULATION_FRAME_Y_OFFSET = 40   # Tab height
SIMULATION_FRAME_WIDTH = WINDOW_WIDTH - SIMULATION_FRAME_X_OFFSET
SIMULATION_FRAME_HEIGHT = WINDOW_HEIGHT - SIMULATION_FRAME_Y_OFFSET