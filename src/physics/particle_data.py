import numpy as np
from pygame import Color


class ParticleData:
    def __init__(self, color):
        self.color = color
        self.max_partices = 5000
        self.positions = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)
