# 2D Chemistry Simulation Game
A powder-game style physics simulation exploring chemical interactions in a 2D world.

## Scientific Foundation: 2D Chemical Physics

### 2D Electron Configuration Theory
In our 2D universe, quantum mechanics operates differently:

1. **Electron Shell Structure**
   - 1st shell (n=1): 2 electrons (1s²)
   - 2nd shell (n=2): 6 electrons (2s², 2p⁴)
   - 3rd shell (n=3): 10 electrons (3s², 3p⁴, 3d⁴)
   - 4th shell (n=4): 14 electrons (4s², 4p⁴, 4d⁴, 4f⁴)

2. **Orbital Theory in 2D**
   - s-orbitals: Circular symmetry (2 electrons)
   - p-orbitals: 2 lobes × 2 orientations (4 electrons)
   - d-orbitals: 4 orientations (4 electrons)
   - f-orbitals: 4 orientations (4 electrons)

3. **Valence Rules**
   - Octet rule becomes "Sextet rule" (6 electrons)
   - Maximum valence electrons: 6
   - Hybridization patterns: sp¹, sp²

### Chemical Bonding in 2D

1. **Covalent Bonding**
   - Single bonds: 2 electrons
   - Double bonds: 4 electrons
   - Triple bonds: 6 electrons (maximum possible)
   - Bond angles: 120° for sp² hybridization

2. **Ionic Bonding**
   - Modified electronegativity scale
   - Maximum charge states: ±3
   - Crystal lattice structures in 2D

3. **Intermolecular Forces**
   - Van der Waals forces
   - 2D hydrogen bonding
   - Dipole-dipole interactions

### Thermodynamics & Kinetics

1. **State Properties**
   - Temperature affects particle velocity
   - Pressure calculated from particle collisions
   - Volume becomes area in 2D

2. **Phase Transitions**
   - Modified phase diagrams
   - Critical points adjusted for 2D
   - Surface tension effects

3. **Reaction Kinetics**
   - Collision theory modified for 2D
   - Activation energy barriers
   - Catalyst behavior

## Technical Implementation Details

### Particle Physics Engine

1. **Force Calculations**

## Element System Design

### Basic Element Properties

1. **Physical Properties**
   - Mass: Affects particle movement and interactions
   - Radius: Determines collision detection area
   - Base Color: Visual representation
   - Phase State: Default state (solid, liquid, gas)
   - Density: Affects stacking and fluid behavior

2. **Chemical Properties**
   - Atomic Number: 1-36 (simplified periodic table)
   - Valence Electrons: 0-6 (following 2D electron configuration)
   - Electronegativity: 0.0-4.0 scale
   - Reactivity: 0-100 scale
   - Oxidation States: Possible charge states in 2D

3. **Behavioral Properties**
   - Temperature Range
     * Melting Point
     * Boiling Point
     * Decomposition Temperature
   - State Changes
     * Phase transition effects
     * Energy absorption/release
   - Special Behaviors
     * Magnetic properties
     * Radioactive decay
     * Catalytic properties

### Element Categories

1. **Basic Elements**
   - 2D-Hydrogen (H): Single electron, highly reactive
   - 2D-Helium (He): Two electrons, inert
   - 2D-Lithium (Li): Single valence electron, metallic
   - 2D-Carbon (C): Forms 3 bonds maximum
   - 2D-Nitrogen (N): Forms 3 bonds with lone pair
   - 2D-Oxygen (O): Forms 2 bonds with 2 lone pairs

2. **Metals**
   - Properties:
     * Positive charge when ionized
     * Electron delocalization
     * Metallic bonding in 2D
   - Examples: Li, Na, K, Fe, Cu, Au

3. **Non-metals**
   - Properties:
     * Negative charge when ionized
     * Covalent bonding
     * Molecular structures
   - Examples: H, C, N, O, F, Cl

4. **Noble Elements**
   - Properties:
     * Full outer shell (6 electrons)
     * Extremely low reactivity
     * No natural bonding
   - Examples: He, Ne, Ar

### Interaction Rules

1. **Bonding Mechanics**
   - Bond Formation
     * Distance threshold for interaction
     * Energy requirements
     * Electron sharing calculations
   - Bond Types
     * Covalent bonds (1-3 pairs)
     * Ionic bonds (charge transfer)
     * Metallic bonds (electron sea)

2. **Reaction Conditions**
   - Temperature Effects
     * Reaction rate modification
     * Bond stability
     * Phase transitions
   - Pressure Effects
     * Collision frequency
     * Reaction probability
     * Compression behavior

3. **Special Interactions**
   - Catalysis
     * Reaction pathway modification
     * Energy barrier reduction
   - Electron Transfer
     * Redox reactions
     * Charge distribution
   - Decomposition
     * Temperature-based
     * Collision-based
     * Radioactive decay

### Implementation Structure

1. **Element Data Format**