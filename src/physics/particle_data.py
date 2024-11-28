import numpy as np
from pygame import Color
import numpy as np

from core.constants import (
    SIMULATION_WIDTH, SIMULATION_HEIGHT
)

class ParticleData:
    class Cell:
        def __init__(self):
            self.max_particles = 500
            self.num_of_particles = 0
            self.positions = np.zeros((self.max_particles, 2), dtype=np.float32)
            self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)

    def __init__(self, color, radius = 2):
        self.color = color
        self.radius = radius
        self.grid_cell_size = radius * 10 * 2
        self.grid_cells_per_row = int((SIMULATION_WIDTH / self.grid_cell_size))
        self.grid_cells_per_col = int((SIMULATION_HEIGHT / self.grid_cell_size))
        self.grid_num_of_cells = self.grid_cells_per_row * self.grid_cells_per_col
        self.grid = [ParticleData.Cell() for _ in range(self.grid_num_of_cells)]
        self.max_particles = 5000
        self.num_of_particles = 0
        self.positions = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)
