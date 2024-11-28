import numpy as np
from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    PARTICLE_SPREAD, VELOCITY_SCALE,
    SIMULATION_FRAME_X_OFFSET, SIMULATION_FRAME_Y_OFFSET,
    SIMULATION_FRAME_WIDTH, SIMULATION_FRAME_HEIGHT
)
from physics.chemical_particle import ChemicalParticle

__all__ = ['ParticleSystemCore']

class ParticleSystemCore:
    def __init__(self):
        """Initialize the particle system core."""
        self.max_particles = 50000
        self.particles_per_frame = PARTICLE_SPREAD
        self.active_particles = 0
        self.element_types = np.empty(self.max_particles, dtype='U1')
        self.chemical_properties = {}
        self._initialize_arrays()
        self._initialize_grid()

    def _initialize_arrays(self):
        """Initialize arrays for particle positions and velocities."""
        self.positions = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.active_mask = np.zeros(self.max_particles, dtype=bool)

    def _initialize_grid(self):
        """Initialize the simulation grid."""
        self.grid_size_x = SIMULATION_FRAME_WIDTH
        self.grid_size_y = SIMULATION_FRAME_HEIGHT
        self.grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=np.int32)

    def create_particle_burst(self, pos, delta_time, element_type='H', burst_size=None):
        """
        Create a burst of particles at a given position.

        Parameters:
            pos (array-like): The (x, y) position where particles are spawned.
            delta_time (float): The time delta since the last update.
            element_type (str): The type of element for the particles.
            burst_size (int, optional): Number of particles to create per burst.
        """
        if burst_size is not None:
            self.particles_per_frame = burst_size
        
        new_count = min(self.particles_per_frame, self.max_particles - self.active_particles)
        if new_count <= 0:
            return

        # Generate random offsets and velocities
        offsets = np.random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD, (new_count, 2))
        new_positions = pos + offsets
        new_velocities = np.random.uniform(-0.5, 0.5, (new_count, 2)) * VELOCITY_SCALE

        # Clip positions to window boundaries
        np.clip(new_positions[:, 0], 0, SIMULATION_FRAME_WIDTH - 1, out=new_positions[:, 0])
        np.clip(new_positions[:, 1], 0, SIMULATION_FRAME_HEIGHT - 1, out=new_positions[:, 1])

        # Update particle arrays
        start_idx = self.active_particles
        end_idx = start_idx + new_count

        self.positions[start_idx:end_idx] = new_positions
        self.velocities[start_idx:end_idx] = new_velocities
        self.element_types[start_idx:end_idx] = element_type
        self.active_particles += new_count

        # Initialize chemical properties for new particles
        for i in range(start_idx, end_idx):
            self.chemical_properties[i] = ChemicalParticle(element_type, i)
