import numpy as np
from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    PARTICLE_SPREAD, VELOCITY_SCALE,
    SIMULATION_WIDTH, SIMULATION_HEIGHT,
    SIMULATION_X_OFFSET, SIMULATION_Y_OFFSET
)
from physics.chemical_particle import ChemicalParticle
from utils.profiler import profile_function

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
        self.grid_size_x = SIMULATION_WIDTH
        self.grid_size_y = SIMULATION_HEIGHT
        self.grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=np.int32)

    @profile_function(threshold_ms=1.0)
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
        np.clip(new_positions[:, 0], 0, WINDOW_WIDTH - 1, out=new_positions[:, 0])
        np.clip(new_positions[:, 1], 0, WINDOW_HEIGHT - 1, out=new_positions[:, 1])

        # Update particle arrays
        start_idx = self.active_particles
        end_idx = start_idx + new_count

        self.positions[start_idx:end_idx] = new_positions
        self.velocities[start_idx:end_idx] = new_velocities
        self.element_types[start_idx:end_idx] = element_type
        self.active_particles += new_count

        # Initialize chemical properties for new particles
        print(f"\n=== Creating Particle Burst ===")
        print(f"Start position: {pos}")
        print(f"Element type: {element_type}")
        print(f"Burst size: {new_count}")
        
        for i in range(start_idx, end_idx):
            print(f"Particle {i} position: {self.positions[i]}")
            self.chemical_properties[i] = ChemicalParticle(element_type, i)
            print(f"Created particle {i} of type {element_type}")
            print(f"Valence electrons: {self.chemical_properties[i].valence_electrons}")
            print(f"Possible bonds: {self.chemical_properties[i].element_data.possible_bonds}")

    def clear_particles(self):
        """Clear all particles from the system."""
        # Before clearing arrays, break all bonds to prevent memory leaks
        for idx in range(self.active_particles):
            if idx in self.chemical_properties:
                self.chemical_properties[idx].break_all_bonds()
        
        self.active_particles = 0
        self.active_mask.fill(False)
        self.chemical_properties.clear()
        # Reset arrays
        self.positions.fill(0)
        self.velocities.fill(0)
        self.element_types.fill('')
        # Reset any GPU resources if they exist
        if hasattr(self, 'physics'):
            if hasattr(self.physics, 'positions_gpu'):
                del self.physics.positions_gpu
            if hasattr(self.physics, 'velocities_gpu'):
                del self.physics.velocities_gpu

    def create_particle(self, position, element_type):
        """Create a new particle"""
        if self.active_particles >= self.max_particles:
            return None
        
        print(f"create_particle called with element_type: {element_type}")
        
        # Validate element_type more strictly
        valid_elements = ['H', 'O', 'N', 'C']
        if not isinstance(element_type, str):
            print(f"Warning: element_type is not a string: {type(element_type)}")
            element_type = 'H'
        elif element_type not in valid_elements:
            print(f"Warning: Invalid element type '{element_type}', valid types are {valid_elements}")
            element_type = 'H'
        
        idx = self.active_particles
        self.positions[idx] = position
        self.velocities[idx] = np.zeros(2)
        self.chemical_properties[idx] = ChemicalParticle(element_type, idx)
        self.element_types[idx] = element_type  # Make sure we're setting this
        self.active_mask[idx] = True
        self.active_particles += 1
        
        print(f"Created particle {idx} of type {element_type}")
        return idx

    def create_particle_burst(self, pos, element_type='H', burst_size=10, spread=20.0, speed=2.0):
        """Create a burst of particles at a given position."""
        # Validate element_type
        valid_elements = ['H', 'O', 'N', 'C']
        if not isinstance(element_type, str) or element_type not in valid_elements:
            element_type = 'H'
        
        try:
            burst_size = int(burst_size)
        except (ValueError, TypeError):
            burst_size = 10

        if self.active_particles + burst_size > self.max_particles:
            burst_size = self.max_particles - self.active_particles
            if burst_size <= 0:
                return

        # Generate random offsets and velocities
        for i in range(burst_size):
            offset = np.random.uniform(-spread, spread, 2)
            new_pos = np.array(pos) + offset
            
            # Clip positions to simulation area
            new_pos[0] = np.clip(new_pos[0], 0, SIMULATION_WIDTH - 1)
            new_pos[1] = np.clip(new_pos[1], 0, SIMULATION_HEIGHT - 1)
            
            # Create particle
            idx = self.create_particle(new_pos, element_type)
            if idx is not None:
                self.velocities[idx] = np.random.uniform(-speed, speed, 2)
