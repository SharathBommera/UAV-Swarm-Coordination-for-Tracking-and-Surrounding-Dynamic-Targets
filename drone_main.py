import pybullet as p
import pybullet_data
import numpy as np
import time

# Parameters
NUM_DRONES = 10
TARGET_POS = np.array([10, 10, 5])
RADIUS = 3
MAX_SPEED = 2.0
SEPARATION_DISTANCE = 1.5
ALIGNMENT_DISTANCE = 2.0
COHESION_DISTANCE = 3.0
ROTOR_THRUST_FACTOR = 0.1

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# Load plane
planeId = p.loadURDF("plane.urdf")

# Load drones
drones = []
for i in range(NUM_DRONES):
    startPos = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    droneId = p.loadURDF("quadrotor.urdf", startPos, startOrientation)
    drones.append(droneId)

# Get rotor joints (assuming 4 rotors per drone)
numJoints = p.getNumJoints(drones[0])
rotorJoints = []
for i in range(numJoints):
    jointInfo = p.getJointInfo(drones[0], i)
    jointName = jointInfo[1].decode("utf-8")
    if "rotor" in jointName:  # Identify rotor joints by name
        rotorJoints.append(i)

# Flocking algorithm
def flocking(droneId, drones, target_pos, radius):
    pos, _ = p.getBasePositionAndOrientation(droneId)
    pos = np.array(pos)
    
    separation = np.zeros(3)
    alignment = np.zeros(3)
    cohesion = np.zeros(3)
    count_sep = 0
    count_ali = 0
    count_coh = 0
    
    for otherId in drones:
        if otherId == droneId:
            continue
        other_pos, _ = p.getBasePositionAndOrientation(otherId)
        other_pos = np.array(other_pos)
        dist = np.linalg.norm(pos - other_pos)
        
        if dist < SEPARATION_DISTANCE:
            separation += (pos - other_pos) / dist
            count_sep += 1
        if dist < ALIGNMENT_DISTANCE:
            alignment += other_pos
            count_ali += 1
        if dist < COHESION_DISTANCE:
            cohesion += other_pos
            count_coh += 1
    
    if count_sep > 0:
        separation /= count_sep
    if count_ali > 0:
        alignment /= count_ali
        alignment = (alignment - pos) / ALIGNMENT_DISTANCE
    if count_coh > 0:
        cohesion /= count_coh
        cohesion = (cohesion - pos) / COHESION_DISTANCE
    
    # Move towards target
    target_dir = target_pos - pos
    target_dist = np.linalg.norm(target_dir)
    if target_dist > radius:
        target_dir = target_dir / target_dist * MAX_SPEED
    else:
        target_dir = np.zeros(3)
    
    # Combine behaviors
    flock_force = separation + alignment + cohesion + target_dir
    flock_force = flock_force / np.linalg.norm(flock_force) * MAX_SPEED
    
    return flock_force

# Main simulation loop
while True:
    for droneId in drones:
        flock_force = flocking(droneId, drones, TARGET_POS, RADIUS)
        
        # Convert flocking force to rotor thrusts
        rotor_thrusts = np.zeros(4)
        rotor_thrusts[0] = max(0, flock_force[0] + flock_force[2])  # Front rotor
        rotor_thrusts[1] = max(0, -flock_force[0] + flock_force[2])  # Back rotor
        rotor_thrusts[2] = max(0, flock_force[1] + flock_force[2])  # Left rotor
        rotor_thrusts[3] = max(0, -flock_force[1] + flock_force[2])  # Right rotor
        
        # Apply rotor thrusts
        for i, joint in enumerate(rotorJoints):
            p.setJointMotorControl2(
                bodyUniqueId=droneId,
                jointIndex=joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=rotor_thrusts[i] * ROTOR_THRUST_FACTOR,
                force=10.0
            )
    
    p.stepSimulation()
    time.sleep(1./240.)

# Disconnect
p.disconnect()