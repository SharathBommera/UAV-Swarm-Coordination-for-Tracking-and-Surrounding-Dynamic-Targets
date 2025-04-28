import numpy as np

# Swarm behavior weights
COHESION_WEIGHT = 0.1
ALIGNMENT_WEIGHT = 0.1
SEPARATION_WEIGHT = 0.2
ALTITUDE_WEIGHT = 0.3  # Increased altitude weight for stability
TARGET_ATTRACTION_WEIGHT = 0.05


def compute_flocking_velocities(positions, velocities, target_position):
    """Compute new velocities for drones based on flocking behavior."""
    num_drones = len(positions)
    new_velocities = np.zeros((num_drones, 3))
    target_altitude = target_position[2]

    for i, pos in enumerate(positions):
        neighbors = [j for j in range(num_drones) if j != i]
        
        if not neighbors:
            new_velocities[i] = velocities[i]
            continue
        
        # Cohesion (Move towards the center of mass)
        center_of_mass = np.mean([positions[j] for j in neighbors], axis=0)
        cohesion = (center_of_mass - pos) * COHESION_WEIGHT

        # Alignment (Match the velocity of neighbors)
        avg_velocity = np.mean([velocities[j] for j in neighbors], axis=0)
        alignment = (avg_velocity - velocities[i]) * ALIGNMENT_WEIGHT

        # Separation (Avoid collisions)
        separation = np.zeros(3)
        for j in neighbors:
            diff = pos - positions[j]
            dist = np.linalg.norm(diff)
            if dist > 0:
                separation += (diff / (dist ** 2))
        separation *= SEPARATION_WEIGHT

        # Move towards target
        target_attraction = (target_position - pos) * TARGET_ATTRACTION_WEIGHT
        
        # Altitude control
        altitude_correction = (target_altitude - pos[2]) * ALTITUDE_WEIGHT
        target_attraction[2] += altitude_correction  # Adjust vertical velocity

        # Compute new velocity
        new_velocity = velocities[i] + cohesion + alignment + separation + target_attraction
        speed = np.linalg.norm(new_velocity)
        if speed > 0:
            new_velocity = new_velocity / speed * min(speed, 1.0)  # Normalize with a max limit
        new_velocities[i] = new_velocity

    return new_velocities
