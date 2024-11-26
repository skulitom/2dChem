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