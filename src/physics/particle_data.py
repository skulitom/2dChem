import numpy as np
from pygame import Color
import numpy as np

from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT
)

class ParticleData:
    class Cell:
        def __init__(self):
            self.max_particles = 5000
            self.num_of_particles = 0
            self.positions = np.zeros((self.max_particles, 2), dtype=np.float32)
            self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)

    def __init__(self, color, radius = 2):
        self.color = color
        self.radius = radius
        self.grid_cell_size = radius * 2 * 2
        self.grid_num_of_cells = int((WINDOW_WIDTH / self.grid_cell_size) * (WINDOW_HEIGHT / self.grid_cell_size))
        #self.grid = np.zeros(self.grid_num_of_cells, dtype=np.float32)
        self.max_particles = 5000
        self.num_of_particles = 0
        self.positions = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)

