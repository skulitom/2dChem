import numpy as np
from core.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    PARTICLE_SPREAD, VELOCITY_SCALE,
    SIMULATION_WIDTH, SIMULATION_HEIGHT,
    SIMULATION_X_OFFSET, SIMULATION_Y_OFFSET
)
from physics.chemical_particle import ChemicalParticle
from utils.profiler import profile_function
import logging

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
        if burst_size is None:
            burst_size = self.particles_per_burst
            
        print(f"\n=== Creating Particle Burst ===")
        print(f"Element type: {element_type}")
        
        # Initialize chemical properties
        for i in range(self.active_particles, self.active_particles + burst_size):
            self.chemical_properties[i] = ChemicalParticle(
                element_type=element_type,
                particle_id=i
            )
            print(f"Created particle {i}")
            print(f"Max bonds: {self.chemical_properties[i].max_bonds}")
            print(f"Possible bonds: {self.chemical_properties[i].element_data.possible_bonds}")

    def clear_particles(self):
        """Clear all particles from the system."""
        # First, break all bonds
        for idx in range(self.active_particles):
            try:
                if idx in self.chemical_properties:
                    particle = self.chemical_properties[idx]
                    if hasattr(particle, 'break_all_bonds'):
                        particle.break_all_bonds()
            except Exception as e:
                logging.warning(f"Error breaking bonds for particle {idx}: {e}")

        # Reset system state
        try:
            self.chemical_properties.clear()
            self.active_particles = 0
            self.active_mask.fill(False)
            self.positions.fill(0)
            self.velocities.fill(0)
            self.element_types.fill('')
        except Exception as e:
            logging.error(f"Error in clear_particles: {e}")
            # Ensure system is in a safe state
            self.active_particles = 0
            self.active_mask.fill(False)
            self.chemical_properties = {}

    def create_particle(self, position, element_type):
        """Create a new particle"""
        if self.active_particles >= self.max_particles:
            return None
        
        try:
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
            
            # Create chemical particle with error handling
            try:
                particle = ChemicalParticle(element_type, idx)
                particle.particle_system = self  # Add reference to particle system
                self.chemical_properties[idx] = particle
            except Exception as e:
                print(f"Error creating chemical particle: {e}")
                return None
            
            self.element_types[idx] = element_type
            self.active_mask[idx] = True
            self.active_particles += 1
            
            print(f"Created particle {idx} of type {element_type}")
            return idx
            
        except Exception as e:
            print(f"Error in create_particle: {e}")
            return None

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

        # Adjust spread based on element type
        if element_type == 'H':
            spread *= 0.7  # Tighter spread for hydrogen
            speed *= 0.8   # Slower speed for hydrogen
        
        # Generate particles in a more organized pattern
        for i in range(burst_size):
            angle = 2 * np.pi * i / burst_size
            radius = np.random.uniform(0, spread)
            offset = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ])
            
            new_pos = (pos[0] + offset[0], pos[1] + offset[1])
            
            # Add some randomness to velocity but maintain general outward direction
            vel_angle = angle + np.random.uniform(-0.2, 0.2)
            velocity = speed * np.array([np.cos(vel_angle), np.sin(vel_angle)])
            
            # Create particle with validated position
            idx = self.create_particle(new_pos, element_type)
            if idx is not None:
                self.velocities[idx] = velocity
