# 2D Chemistry Simulation Game

A powder-game style physics simulation exploring chemical interactions in a two-dimensional universe.

## Scientific Foundation: 2D Chemical Physics

In a two-dimensional (2D) universe, fundamental physical laws such as quantum mechanics and electromagnetism still govern the behavior of particles, but their manifestations differ due to reduced dimensionality. Electrons orbit atomic nuclei, forming atoms and molecules, but the structures of electron orbitals, energy levels, and the periodic table are distinct from those in our three-dimensional (3D) world.

### Electromagnetic Force in 2D

In 3D space, the electromagnetic force between two point charges follows Coulomb's Law, which states that the force is inversely proportional to the square of the distance between them (1/r²). This is because the electric field spreads out over the surface area of a sphere, scaling with r².

In 2D space, the electric field spreads over the circumference of a circle, which scales linearly with r. Consequently, the electromagnetic force in 2D follows an inverse linear law (1/r). This fundamental difference significantly alters the potential energy of electrons in atoms and the spacing of energy levels.

## 2D Quantum Mechanics and Electron Orbitals

### Quantum Numbers in 2D

- **Principal Quantum Number (n)**: Indicates the energy level or shell of an electron, with n = 1, 2, 3, ...
- **Angular Momentum Quantum Number (l)**: In 2D, l can take integer values from 0 up to (n - 1).
- **Magnetic Quantum Number (mₛ)**: In 2D, for each l > 0, there are only two possible orientations corresponding to the electron orbiting clockwise or counterclockwise (mₛ = ±l).
- **Spin Quantum Number (mₛ)**: Electrons have spin quantum numbers of +½ or -½.

### Electron Orbitals in 2D

- **s Orbitals (l = 0)**:
  - Shape: Circular symmetry.
  - Number of Orbitals: 1.
  - Electrons per Orbital: 2 (due to spin).
- **p, d, f, ... Orbitals (l > 0)**:
  - For each l > 0, there are 2 orbitals (mₛ = +l and mₛ = -l).
  - Electrons per Orbital: 2.
  - Total Electrons per l Level: 2 orbitals × 2 electrons = 4 electrons.

### Electron Shell Structure in 2D

- **First Shell (n = 1)**:
  - l = 0.
  - Total Electrons: 1 orbital × 2 electrons = 2 electrons.
- **Second Shell (n = 2)**:
  - l = 0 and l = 1.
  - Total Electrons:
    - l = 0: 1 orbital × 2 electrons = 2 electrons.
    - l = 1: 2 orbitals × 2 electrons = 4 electrons.
  - **Total for n = 2**: 6 electrons.
- **Third Shell (n = 3)**:
  - l = 0, 1, 2.
  - Total Electrons:
    - l = 0: 2 electrons.
    - l = 1: 4 electrons.
    - l = 2: 4 electrons.
  - **Total for n = 3**: 10 electrons.
- **Fourth Shell (n = 4)**:
  - l = 0, 1, 2, 3.
  - Total Electrons:
    - l = 0: 2 electrons.
    - l = 1: 4 electrons.
    - l = 2: 4 electrons.
    - l = 3: 4 electrons.
  - **Total for n = 4**: 14 electrons.

### Filling Order of Orbitals

In 2D, the energy levels of orbitals depend on both n and l, but the change in the electromagnetic force law (from 1/r² to 1/r) alters the energy ordering compared to 3D. Generally, energy increases with n and l, but specific calculations are required to determine the exact order in which orbitals fill.

## Valence Electrons and the "Sextet Rule"

- **Valence Electrons**: Electrons in the outermost shell (highest n) determine the chemical properties of an element.
- **Sextet Rule**: Atoms tend to achieve six valence electrons in their outer shell for stability, analogous to the octet rule in 3D. This is due to the maximum of six electrons in the second shell (n = 2).

## The 2D Periodic Table

Due to differences in electron orbital structures and energy levels, the periodic table in 2D has a unique arrangement. Periods correspond to the filling of electron shells, and groups contain elements with similar valence electron configurations and chemical properties.

### Notable Differences from the 3D Periodic Table

- **Noble Gases**: Elements with a full valence shell of six electrons are noble gases in 2D (e.g., elements with 2, 6, 12, 20 electrons).
- **Halogens**: Elements with five valence electrons, needing one more to complete the sextet, are highly reactive non-metals.
- **Alkali Metals**: Elements with one valence electron are highly reactive metals.
- **Transition Metals**: The filling of d and higher orbitals differs, leading to variations in the transition metal series.

## Chemical Bonding in 2D

### Covalent Bonding

- **Electron Sharing**: Atoms share electrons to achieve a full sextet in their valence shell.
- **Bond Types**:
  - **Single Bonds**: Sharing one pair of electrons (2 electrons).
  - **Double Bonds**: Sharing two pairs of electrons (4 electrons).
  - **Triple Bonds**: Sharing three pairs of electrons (6 electrons), the maximum due to the sextet rule.
- **Molecular Geometry**:
  - **Planar Structures**: Molecules are inherently planar in 2D, influencing bond angles and shapes.
  - **Bond Angles**: Determined by hybridization; for example, sp² hybridization results in 120° angles.

### Ionic Bonding

- **Electron Transfer**: Atoms with low electronegativity lose electrons to atoms with high electronegativity, forming ions.
- **Electrostatic Attraction**: Positive and negative ions attract, forming ionic compounds.
- **2D Lattice Structures**: Ionic compounds arrange in planar crystal lattices.

### Metallic Bonding

- **Delocalized Electrons**: Valence electrons are shared among all metal atoms, allowing for conductivity.
- **Planar Metal Structures**: Metals form continuous 2D sheets.

### Intermolecular Forces

- **Dipole-Dipole Interactions**: Occur between polar molecules.
- **Van der Waals Forces**: Weak attractions due to temporary dipoles.
- **Hydrogen Bonding**: Strong dipole interactions involving hydrogen bonded to electronegative atoms.

## Thermodynamics and Kinetics in 2D

### State Properties

- **Temperature**: Measures average kinetic energy in the 2D plane.
- **Pressure**: Force per unit length in 2D (since area reduces to length).
- **Area Instead of Volume**: Thermodynamic equations adjust for two dimensions.

### Phase Transitions

- **Modified Phase Diagrams**: Reflect 2D pressure and temperature relationships.
- **Critical Points**: Adjusted due to changes in intermolecular forces and dimensionality.

### Reaction Kinetics

- **Collision Theory**: In 2D, particles collide differently, affecting reaction rates.
- **Activation Energy**: Minimum energy required for a reaction to occur.
- **Catalysts**: Lower activation energy, increasing reaction rates without being consumed.

## Technical Implementation Details

### Particle Physics Engine

- **Force Calculations**: Simulate interactions based on the 1/r electromagnetic force law specific to 2D space.
- **Quantum Mechanics Approximation**: Simplify quantum behaviors to create a playable simulation while retaining scientific accuracy.
- **Collision Detection**: Implement efficient algorithms for detecting and handling collisions in a 2D environment.

### Simulation Mechanics

- **Planar Dynamics**: All particle motions and interactions occur within a two-dimensional plane.
- **Visual Representation**: Use colors, shapes, and animations to represent different elements and compounds.
- **User Interaction**: Players can introduce elements, adjust environmental conditions, and observe resulting behaviors.

## Element System Design

### Basic Element Properties

- **Physical Properties**
  - **Mass**: Determines inertia and response to forces.
  - **Radius**: Affects collision detection and interaction range.
  - **Color**: Visual identifier for elements and compounds.
  - **Phase State**: Default state under standard conditions (solid, liquid, gas).
  - **Density**: Influences buoyancy and layering in fluids.
- **Chemical Properties**
  - **Atomic Number**: Number of protons/electrons defining the element.
  - **Electron Configuration**: Based on 2D orbital structures.
  - **Valence Electrons**: Up to 6 in the outermost shell.
  - **Electronegativity**: Ability to attract electrons in a bond.
  - **Reactivity**: Likelihood to participate in chemical reactions.
  - **Oxidation States**: Possible ionic charges.
- **Behavioral Properties**
  - **Temperature Ranges**:
    - **Melting Point**
    - **Boiling Point**
    - **Sublimation Point**
  - **State Changes**: Energy absorption or release during phase transitions.
  - **Special Behaviors**:
    - **Magnetism**
    - **Radioactivity**
    - **Catalytic Activity**

### Element Categories

- **Noble Gases**
  - **Properties**: Full valence shell (6 electrons), chemically inert.
  - **Examples**: Elements with electron configurations ending in a filled shell.
- **Halogens**
  - **Properties**: Five valence electrons, highly reactive, form negative ions.
  - **Examples**: Elements seeking one electron to complete their sextet.
- **Alkali Metals**
  - **Properties**: One valence electron, highly reactive, form positive ions.
  - **Examples**: Elements readily losing an electron to achieve a full inner shell.
- **Transition Metals**
  - **Properties**: Involve filling of l > 1 orbitals, exhibit multiple oxidation states.
  - **2D Considerations**: Adjusted orbital filling due to altered energy levels.

## Interaction Rules

### Bonding Mechanics

- **Formation Criteria**:
  - **Proximity**: Atoms must be close enough to interact.
  - **Energy**: Sufficient energy must be available for bond formation.
  - **Electron Configuration**: Determines bonding capacity.
- **Bond Types**:
  - **Covalent Bonds**: Electron sharing between non-metals.
  - **Ionic Bonds**: Electron transfer between metals and non-metals.
  - **Metallic Bonds**: Delocalized electrons among metal atoms.

### Reaction Conditions

- **Temperature Effects**:
  - **Reaction Rates**: Increase with temperature due to higher kinetic energy.
  - **Phase Changes**: Substances may change states with temperature variations.
- **Pressure Effects**:
  - **Collision Frequency**: Higher pressure increases collision rates.
  - **Reaction Equilibrium**: Can shift with changes in pressure.

### Special Interactions

- **Catalysis**:
  - **Mechanism**: Catalysts provide alternative pathways with lower activation energy.
- **Redox Reactions**:
  - **Electron Transfer**: Involves oxidation (loss) and reduction (gain) of electrons.
- **Decomposition and Decay**:
  - **Thermal Decomposition**: Breakdown due to heat.
  - **Radioactivity**: Unstable nuclei emitting particles.

## Implementation Structure

### Element Data Format

- **Data Fields**:
  - **Physical Properties**: Mass, radius, density, phase state, color.
  - **Chemical Properties**: Atomic number, electron configuration, valence electrons, electronegativity, oxidation states.
  - **Behavioral Properties**: Temperature thresholds, reactivity, special behaviors.
- **Storage**: Elements are defined in data files or databases for easy access and modification.

### Interaction Algorithms

- **Forces and Movements**: Simulate electromagnetic forces, gravity (if applicable), and particle dynamics.
- **Chemical Reactions**: Implement rules for bond formation and breaking based on element properties and environmental conditions.
- **State Changes**: Handle phase transitions and associated energy changes.

