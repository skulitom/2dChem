def _try_form_bond(system, idx1, idx2, distance) -> bool:
    """Try to form a bond between two particles"""
    chem1 = system.chemical_properties[idx1]
    chem2 = system.chemical_properties[idx2]
    
    # Check if particles are already bonded
    if any(bond.particle_id == idx2 for bond in chem1.bonds):
        return False
    
    # Create and add bonds with proper bond type determination
    bond_type = 'covalent'  # Default to covalent
    bond_strength = 1.0
    
    # Determine bond type based on electronegativity difference
    en_diff = abs(chem1.element_data.electronegativity - chem2.element_data.electronegativity)
    if en_diff > 1.7:
        bond_type = 'ionic'
        bond_strength = 0.8
    elif en_diff > 0.4:
        bond_type = 'polar_covalent'
        bond_strength = 1.2
    
    bond1 = Bond(
        particle_id=idx2,
        bond_type=bond_type,
        strength=bond_strength,
        shared_electrons=2
    )
    bond2 = Bond(
        particle_id=idx1,
        bond_type=bond_type,
        strength=bond_strength,
        shared_electrons=2
    )
    
    # Add bonds to both particles
    chem1.bonds.append(bond1)
    chem2.bonds.append(bond2)
    
    # Reduce velocities significantly after bonding
    avg_velocity = (system.velocities[idx1] + system.velocities[idx2]) * 0.5
    system.velocities[idx1] = avg_velocity * 0.7
    system.velocities[idx2] = avg_velocity * 0.7
    
    return True 