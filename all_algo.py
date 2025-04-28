import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from collections import deque
import threading

class UAVSwarmEnvMulti:
    def __init__(self):
        # Connect to PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        # Create separate spaces for each algorithm
        # Each algorithm will have its own region in the simulation
        self.regions = {
            "ABC": {"offset": np.array([-10, 10, 0]), "boundary": np.array([10, 10, 10])},
            "PSO": {"offset": np.array([10, 10, 0]), "boundary": np.array([10, 10, 10])},
            "BOIDS": {"offset": np.array([-10, -10, 0]), "boundary": np.array([10, 10, 10])},
            "LEADER": {"offset": np.array([10, -10, 0]), "boundary": np.array([10, 10, 10])}
        }
        
        # Create environments for each algorithm
        self.envs = {}
        self.threads = {}
        
        # Create quadrants
        self._create_quadrants()
        
        # Add labels for each algorithm
        self._add_labels()
        
    def _create_quadrants(self):
        # Create a plane for the base
        p.loadURDF("plane.urdf")
        
        # Create boundary visual indicators for each quadrant
        for region_name, region_data in self.regions.items():
            offset = region_data["offset"]
            size = region_data["boundary"]
            
            # Draw boundary lines for each quadrant
            line_width = 3
            color = [1, 1, 1]  # White boundaries
            
            # Bottom rectangle
            p.addUserDebugLine([offset[0] - size[0], offset[1] - size[1], 0], 
                              [offset[0] + size[0], offset[1] - size[1], 0], 
                              color, lineWidth=line_width)
            p.addUserDebugLine([offset[0] + size[0], offset[1] - size[1], 0], 
                              [offset[0] + size[0], offset[1] + size[1], 0], 
                              color, lineWidth=line_width)
            p.addUserDebugLine([offset[0] + size[0], offset[1] + size[1], 0], 
                              [offset[0] - size[0], offset[1] + size[1], 0], 
                              color, lineWidth=line_width)
            p.addUserDebugLine([offset[0] - size[0], offset[1] + size[1], 0], 
                              [offset[0] - size[0], offset[1] - size[1], 0], 
                              color, lineWidth=line_width)
            
            # Top rectangle at height 10
            p.addUserDebugLine([offset[0] - size[0], offset[1] - size[1], size[2]], 
                              [offset[0] + size[0], offset[1] - size[1], size[2]], 
                              color, lineWidth=line_width)
            p.addUserDebugLine([offset[0] + size[0], offset[1] - size[1], size[2]], 
                              [offset[0] + size[0], offset[1] + size[1], size[2]], 
                              color, lineWidth=line_width)
            p.addUserDebugLine([offset[0] + size[0], offset[1] + size[1], size[2]], 
                              [offset[0] - size[0], offset[1] + size[1], size[2]], 
                              color, lineWidth=line_width)
            p.addUserDebugLine([offset[0] - size[0], offset[1] + size[1], size[2]], 
                              [offset[0] - size[0], offset[1] - size[1], size[2]], 
                              color, lineWidth=line_width)
            
            # Connecting lines
            p.addUserDebugLine([offset[0] - size[0], offset[1] - size[1], 0], 
                              [offset[0] - size[0], offset[1] - size[1], size[2]], 
                              color, lineWidth=line_width)
            p.addUserDebugLine([offset[0] + size[0], offset[1] - size[1], 0], 
                              [offset[0] + size[0], offset[1] - size[1], size[2]], 
                              color, lineWidth=line_width)
            p.addUserDebugLine([offset[0] + size[0], offset[1] + size[1], 0], 
                              [offset[0] + size[0], offset[1] + size[1], size[2]], 
                              color, lineWidth=line_width)
            p.addUserDebugLine([offset[0] - size[0], offset[1] + size[1], 0], 
                              [offset[0] - size[0], offset[1] + size[1], size[2]], 
                              color, lineWidth=line_width)
    
    def _add_labels(self):
        for region_name, region_data in self.regions.items():
            offset = region_data["offset"]
            p.addUserDebugText(
                f"{region_name} Algorithm", 
                [offset[0], offset[1], 11], 
                textColorRGB=[1, 1, 1],
                textSize=1.5
            )
    
    def run_all(self):
        # Initialize environments
        self.envs["ABC"] = UAVSwarmEnvABC(self.regions["ABC"]["offset"])
        self.envs["PSO"] = UAVSwarmEnvPSO(self.regions["PSO"]["offset"])
        self.envs["BOIDS"] = UAVSwarmEnvBoids(self.regions["BOIDS"]["offset"])
        self.envs["LEADER"] = UAVSwarmEnvLeaderFollower(self.regions["LEADER"]["offset"])
        
        # Reset all environments
        for env_name, env in self.envs.items():
            env.reset()
        
        # Run simulation
        try:
            while True:
                # Step all environments
                for env_name, env in self.envs.items():
                    action = np.zeros((env.num_drones, 3))  # No external actions needed
                    env.step(action)
                
                # Step simulation once per loop iteration
                p.stepSimulation()
                
                # Control simulation speed
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Simulation ended by user")
            
        # Close all environments
        for env_name, env in self.envs.items():
            env.close()
        
        p.disconnect()

# ABC Algorithm Implementation
class UAVSwarmEnvABC:
    def __init__(self, offset):
        self.offset = offset  # Position offset for this algorithm
        
        # ABC algorithm parameters
        self.num_drones = 5  # Total number of drones in the swarm
        self.employed_bees = 3  # 3 employed bees
        self.onlooker_bees = self.num_drones - self.employed_bees  # 2 onlooker bees
        self.food_sources = self.employed_bees  # Number of food sources equals employed bees
        self.limit = 5  # Limit before scout bee phase
        self.swarm_speed = 0.05
        self.boundary = np.array([10, 10, 10], dtype=np.float32)  # 10x10x10 space
        
        # Path trailing parameters
        self.trail_length = 50  # Number of positions to keep in the trail
        self.line_ids = [[] for _ in range(self.num_drones)]  # Store line IDs for each drone
        self.position_history = [deque(maxlen=self.trail_length) for _ in range(self.num_drones)]
        
        # Target parameters - ensure free movement within the space
        self.target_speed = 0.04
        self.target_position = np.array([5, 5, 5], dtype=np.float32)  # Start in the middle
        self.target_velocity = np.random.uniform(-self.target_speed, self.target_speed, size=3).astype(np.float32)
        
        # Initialize target drone
        self.target_drone = self._add_target_drone()
        
        # Initialize food sources (potential solutions)
        self.food_positions = np.random.rand(self.food_sources, 3) * self.boundary
        self.food_qualities = np.zeros(self.food_sources)
        self.trial_counter = np.zeros(self.food_sources)
        
        # Initialize quadrotors
        self.drones = []
        self.drone_positions = np.random.rand(self.num_drones, 3) * self.boundary
        self.drone_velocities = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.drone_colors = []
        
        # Create quadrotors with different colors for employed and onlooker bees
        for i in range(self.num_drones):
            if i < self.employed_bees:
                # Employed bees (blue)
                color = [0, 0, 1, 1]
            else:
                # Onlooker bees (green)
                color = [0, 1, 0, 1]
            
            self.drone_colors.append(color)
            drone_id = self._create_quadrotor(self.drone_positions[i] + self.offset, color)
            self.drones.append(drone_id)
            
            # Initialize position history for trail
            self.position_history[i].append(self.drone_positions[i].copy())

    def _create_quadrotor(self, position, color):
        """Create a quadrotor at the specified position."""
        quadrotor_id = p.loadURDF("quadrotor.urdf", position)
        
        # Change the color of the quadrotor
        for link_idx in range(p.getNumJoints(quadrotor_id) + 1):
            p.changeVisualShape(quadrotor_id, link_idx - 1, rgbaColor=color)
            
        return quadrotor_id
   
    def _add_target_drone(self):
        """Create the target drone (red sphere)."""
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
        return p.createMultiBody(baseMass=1, baseVisualShapeIndex=visual_shape_id, basePosition=self.target_position + self.offset)
   
    def _update_target_position(self):
        """Move the target freely within the 10x10x10 space."""
        # Add small random variations to velocity
        self.target_velocity += np.random.uniform(-0.01, 0.01, size=3).astype(np.float32)
        
        # Normalize and scale velocity to maintain consistent speed
        self.target_velocity = self.target_velocity / (np.linalg.norm(self.target_velocity) + 1e-5) * self.target_speed
        
        # Update position
        self.target_position += self.target_velocity
        
        # Bounce off boundaries to keep within the 10x10x10 space
        for i in range(3):
            if self.target_position[i] <= 0 or self.target_position[i] >= self.boundary[i]:
                self.target_velocity[i] *= -1
                self.target_position[i] = np.clip(self.target_position[i], 0, self.boundary[i])
        
        # Update target position in simulation
        p.resetBasePositionAndOrientation(self.target_drone, self.target_position + self.offset, [0, 0, 0, 1])
   
    def _evaluate_food_source(self, position):
        """Evaluate the quality of a food source based on distance to target."""
        distance = np.linalg.norm(position - self.target_position)
        return 1.0 / (distance + 1e-5)  # Higher quality for closer positions
   
    def _employed_bee_phase(self):
        """Employed bees phase of ABC algorithm."""
        for i in range(self.employed_bees):
            # Create a new solution near the food source
            new_position = self.food_positions[i].copy()
            
            # Choose a random dimension to modify
            dim = np.random.randint(0, 3)
            
            # Choose a random other food source
            k = np.random.randint(0, self.food_sources)
            while k == i:
                k = np.random.randint(0, self.food_sources)
            
            # Create new solution
            phi = np.random.uniform(-1, 1)
            new_position[dim] = new_position[dim] + phi * (new_position[dim] - self.food_positions[k][dim])
            
            # Ensure within boundaries
            new_position = np.clip(new_position, 0, self.boundary)
            
            # Evaluate new solution
            new_quality = self._evaluate_food_source(new_position)
            
            # Greedy selection
            if new_quality > self.food_qualities[i]:
                self.food_positions[i] = new_position
                self.food_qualities[i] = new_quality
                self.trial_counter[i] = 0
            else:
                self.trial_counter[i] += 1
   
    def _calculate_probabilities(self):
        """Calculate selection probabilities for onlooker bees."""
        total_fitness = np.sum(self.food_qualities) + 1e-10
        return self.food_qualities / total_fitness
   
    def _onlooker_bee_phase(self):
        """Onlooker bees phase of ABC algorithm."""
        probabilities = self._calculate_probabilities()
        
        for i in range(self.employed_bees, self.num_drones):
            # Select a food source based on probability
            selected_source = np.random.choice(self.food_sources, p=probabilities)
            
            # Create a new solution near the selected food source
            new_position = self.food_positions[selected_source].copy()
            
            # Choose a random dimension to modify
            dim = np.random.randint(0, 3)
            
            # Choose a random other food source
            k = np.random.randint(0, self.food_sources)
            while k == selected_source:
                k = np.random.randint(0, self.food_sources)
            
            # Create new solution
            phi = np.random.uniform(-1, 1)
            new_position[dim] = new_position[dim] + phi * (new_position[dim] - self.food_positions[k][dim])
            
            # Ensure within boundaries
            new_position = np.clip(new_position, 0, self.boundary)
            
            # Evaluate new solution
            new_quality = self._evaluate_food_source(new_position)
            
            # Greedy selection
            if new_quality > self.food_qualities[selected_source]:
                self.food_positions[selected_source] = new_position
                self.food_qualities[selected_source] = new_quality
                self.trial_counter[selected_source] = 0
            else:
                self.trial_counter[selected_source] += 1
   
    def _scout_bee_phase(self):
        """Scout bees phase of ABC algorithm."""
        for i in range(self.food_sources):
            if self.trial_counter[i] >= self.limit:
                # Abandon the food source and create a new random one
                self.food_positions[i] = np.random.rand(3) * self.boundary
                self.food_qualities[i] = self._evaluate_food_source(self.food_positions[i])
                self.trial_counter[i] = 0
   
    def _update_drone_positions(self):
        """Update drone positions based on the ABC algorithm roles."""
        # Update employed bees - they move toward their food sources
        for i in range(self.employed_bees):
            direction = self.food_positions[i] - self.drone_positions[i]
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:  # Only move if not very close to the food source
                self.drone_velocities[i] = direction / distance * self.swarm_speed
            else:
                self.drone_velocities[i] = np.zeros(3)
            
            self.drone_positions[i] += self.drone_velocities[i]
        
        # Update onlooker bees - they move based on the food sources they selected
        probabilities = self._calculate_probabilities()
        for i in range(self.employed_bees, self.num_drones):
            # Select a food source based on probability
            selected_source = np.random.choice(self.food_sources, p=probabilities)
            
            direction = self.food_positions[selected_source] - self.drone_positions[i]
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                self.drone_velocities[i] = direction / distance * self.swarm_speed
            else:
                self.drone_velocities[i] = np.zeros(3)
            
            self.drone_positions[i] += self.drone_velocities[i]
   
    def _update_trails(self):
        """Update the trail lines for each drone."""
        # First remove old lines
        for drone_lines in self.line_ids:
            for line_id in drone_lines:
                p.removeUserDebugItem(line_id)
        
        # Clear line lists
        self.line_ids = [[] for _ in range(self.num_drones)]
        
        # Add latest positions to history
        for i in range(self.num_drones):
            self.position_history[i].append(self.drone_positions[i].copy())
        
        # Draw new lines
        for i in range(self.num_drones):
            positions = list(self.position_history[i])
            if len(positions) < 2:
                continue
                
            for j in range(len(positions) - 1):
                line_id = p.addUserDebugLine(
                    lineFromXYZ=positions[j] + self.offset,
                    lineToXYZ=positions[j + 1] + self.offset,
                    lineColorRGB=self.drone_colors[i][:3],  # Use drone color for line
                    lineWidth=1.0,
                    lifeTime=0  # Persistent until removed
                )
                self.line_ids[i].append(line_id)
   
    def step(self, action):
        """Update environment for one step using ABC algorithm."""
        # 1. Update target position
        self._update_target_position()
        
        # 2. Evaluate all food sources
        for i in range(self.food_sources):
            self.food_qualities[i] = self._evaluate_food_source(self.food_positions[i])
        
        # 3. Run ABC algorithm phases
        self._employed_bee_phase()
        self._onlooker_bee_phase()
        self._scout_bee_phase()
        
        # 4. Update drone positions
        self._update_drone_positions()
        
        # 5. Update trails
        self._update_trails()
        
        # 6. Update drone positions in simulation
        for i in range(self.num_drones):
            # Get current orientation
            _, orientation = p.getBasePositionAndOrientation(self.drones[i])
            
            # Reset position (and orientation if needed)
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i] + self.offset, orientation)
        
        # Calculate mean distance to target for reward
        mean_distance = np.mean([np.linalg.norm(self.drone_positions[i] - self.target_position)
                              for i in range(self.num_drones)])
        
        reward = -mean_distance
        done = False
        
        return self._get_observation(), reward, done, {}
   
    def reset(self):
        """Reset the environment."""
        # Reset target
        self.target_position = np.array([5, 5, 5], dtype=np.float32)
        self.target_velocity = np.random.uniform(-self.target_speed, self.target_speed, size=3).astype(np.float32)
        
        # Reset food sources
        self.food_positions = np.random.rand(self.food_sources, 3) * self.boundary
        self.food_qualities = np.zeros(self.food_sources)
        self.trial_counter = np.zeros(self.food_sources)
        
        # Reset drones
        self.drone_positions = np.random.rand(self.num_drones, 3) * self.boundary
        self.drone_velocities = np.zeros((self.num_drones, 3), dtype=np.float32)
        
        # Reset position history and clear trails
        for i in range(self.num_drones):
            self.position_history[i].clear()
            self.position_history[i].append(self.drone_positions[i].copy())
            
            for line_id in self.line_ids[i]:
                p.removeUserDebugItem(line_id)
        
        self.line_ids = [[] for _ in range(self.num_drones)]
        
        # Reset drone positions in simulation
        for i in range(self.num_drones):
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i] + self.offset, [0, 0, 0, 1])
        
        # Reset target drone
        p.resetBasePositionAndOrientation(self.target_drone, self.target_position + self.offset, [0, 0, 0, 1])
        
        return self._get_observation()
   
    def _get_observation(self):
        """Return the observation."""
        return {
            'target_position': self.target_position,
            'drone_positions': self.drone_positions,
            'food_positions': self.food_positions,
            'food_qualities': self.food_qualities
        }
   
    def render(self, mode='human'):
        """Rendering is handled by PyBullet."""
        pass
   
    def close(self):
        """Clean up debug visualizations"""
        for drone_lines in self.line_ids:
            for line_id in drone_lines:
                p.removeUserDebugItem(line_id)

# PSO Algorithm Implementation
class UAVSwarmEnvPSO:
    def __init__(self, offset):
        self.offset = offset
        self.num_drones = 5
        self.swarm_speed = 0.05
        self.target_speed = 0.03
        self.boundary = np.array([10, 10, 10])
        self.path_colors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]
        ]
        self.previous_positions = []
        self.line_ids = [[] for _ in range(self.num_drones)]

        self.target_position = np.random.rand(3) * 5 + np.array([3, 3, 3])
        self.target_velocity = np.random.randn(3) * self.target_speed
        self.target_drone = self._add_target_drone()

        self.drones = []
        self.drone_positions = np.random.rand(self.num_drones, 3) * 5
        self.drone_velocities = np.random.randn(self.num_drones, 3) * self.swarm_speed
        self.personal_best_positions = np.copy(self.drone_positions)
        self.global_best_position = np.mean(self.drone_positions, axis=0)

        for i in range(self.num_drones):
            drone_id = p.loadURDF("quadrotor.urdf", self.drone_positions[i] + self.offset)
            self.drones.append(drone_id)
            self.previous_positions.append(self.drone_positions[i].copy())

    def _add_target_drone(self):
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=self.target_position + self.offset)

    def _update_target_position(self):
        self.target_position += self.target_velocity
        for i in range(3):
            if self.target_position[i] < 0 or self.target_position[i] > self.boundary[i]:
                self.target_velocity[i] *= -1
        self.target_velocity += np.random.uniform(-0.01, 0.01, size=3)
        self.target_velocity = np.clip(self.target_velocity, -self.target_speed, self.target_speed)
        p.resetBasePositionAndOrientation(self.target_drone, self.target_position + self.offset, [0, 0, 0, 1])

    def _pso_behavior(self):
        inertia_weight = 0.5
        cognitive_coeff = 1.5
        social_coeff = 1.5
        separation_coeff = 1.0
        max_velocity = 0.1
        new_velocities = (
            inertia_weight * self.drone_velocities
            + cognitive_coeff * np.random.rand(self.num_drones, 3) * (self.personal_best_positions - self.drone_positions)
            + social_coeff * np.random.rand(self.num_drones, 3) * (self.global_best_position - self.drone_positions)
        )

        for i in range(self.num_drones):
            repulsion = np.zeros(3)
            for j in range(self.num_drones):
                if i != j:
                    dist_vec = self.drone_positions[i] - self.drone_positions[j]
                    dist = np.linalg.norm(dist_vec)
                    if dist < 1.0 and dist > 1e-2:
                        repulsion += dist_vec / (dist ** 2)
            new_velocities[i] += separation_coeff * repulsion

        new_velocities = np.clip(new_velocities, -max_velocity, max_velocity)
        new_positions = self.drone_positions + new_velocities
        new_positions = np.clip(new_positions, 0, self.boundary)

        for i in range(self.num_drones):
            if np.linalg.norm(new_positions[i] - self.target_position) < np.linalg.norm(self.personal_best_positions[i] - self.target_position):
                self.personal_best_positions[i] = new_positions[i]
                
        # Update global best position
        best_drone = np.argmin([np.linalg.norm(self.personal_best_positions[i] - self.target_position) for i in range(self.num_drones)])
        self.global_best_position = self.personal_best_positions[best_drone]

        return new_velocities

    def step(self, action):
        # Clear previous lines
        for i in range(self.num_drones):
            for line_id in self.line_ids[i]:
                p.removeUserDebugItem(line_id)
            self.line_ids[i] = []
            
        self._update_target_position()
        self.drone_velocities = self._pso_behavior()
        self.drone_positions += self.drone_velocities

        for i in range(self.num_drones):
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i] + self.offset, [0, 0, 0, 1])
            
            # Draw trail lines
            line_id = p.addUserDebugLine(
                self.previous_positions[i] + self.offset,
                self.drone_positions[i] + self.offset,
                self.path_colors[i % len(self.path_colors)],
                1,
                lifeTime=15
            )
            self.line_ids[i].append(line_id)
            self.previous_positions[i] = self.drone_positions[i].copy()

        return self._get_observation(), self._compute_reward(), False, {}

    def reset(self):
        self.target_position = np.random.rand(3) * 5 + np.array([3, 3, 3])
        self.target_velocity = np.random.randn(3) * self.target_speed
        self.drone_positions = np.random.rand(self.num_drones, 3) * 5
        self.drone_velocities = np.random.randn(self.num_drones, 3) * self.swarm_speed
        self.personal_best_positions = np.copy(self.drone_positions)
        self.global_best_position = np.mean(self.drone_positions, axis=0)
        self.previous_positions = [pos.copy() for pos in self.drone_positions]

        # Reset target
        p.resetBasePositionAndOrientation(self.target_drone, self.target_position + self.offset, [0, 0, 0, 1])
        
        # Reset drones
        for i in range(self.num_drones):
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i] + self.offset, [0, 0, 0, 1])
            
            # Clear lines
            for line_id in self.line_ids[i]:
                p.removeUserDebugItem(line_id)
            self.line_ids[i] = []

        return self._get_observation()

    def _get_observation(self):
        return np.hstack([self.target_position, self.drone_positions.flatten()])

    def _compute_reward(self):
        distance_to_target = np.mean([np.linalg.norm(self.drone_positions[i] - self.target_position) for i in range(self.num_drones)])
        return -distance_to_target

    def close(self):
        for i in range(self.num_drones):
            for line_id in self.line_ids[i]:
                p.removeUserDebugItem(line_id)

# Boids Algorithm Implementation
class UAVSwarmEnvBoids:
    def __init__(self, offset):
        self.offset = offset
        self.num_drones = 5
        self.swarm_speed = 0.05
        self.boundary = np.array([10, 10, 10])
        
        # Boids algorithm parameters
        self.perception_radius = 3.0
        self.separation_weight = 1.5
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        self.target_weight = 2.0  # Weight for target attraction
        self.max_speed = 0.1
        self.min_speed = 0.01
        
        # Path trail parameters
        self.trail_length = 50
        self.line_ids = [[] for _ in range(self.num_drones)]
        self.position_history = [deque(maxlen=self.trail_length) for _ in range(self.num_drones)]
        
        # Initialize target
        self.target_speed = 0.03
        self.target_position = np.array([5, 5, 5], dtype=np.float32)
        self.target_velocity = np.random.uniform(-self.target_speed, self.target_speed, size=3).astype(np.float32)
        self.target_drone = self._add_target_drone()
        
        # Initialize drones with random positions and velocities
        self.drones = []
        self.drone_positions = np.random.rand(self.num_drones, 3) * self.boundary * 0.8 + self.boundary * 0.1
        self.drone_velocities = np.random.randn(self.num_drones, 3) * 0.01
        
        # Create drones with different colors
        for i in range(self.num_drones):
            # Use a gradient of blue colors
            color = [0, 0.5 + 0.5 * (i / self.num_drones), 1, 1]
            drone_id = self._create_quadrotor(self.drone_positions[i] + self.offset, color)
            self.drones.append(drone_id)
            self.position_history[i].append(self.drone_positions[i].copy())
    
    def _create_quadrotor(self, position, color):
        """Create a quadrotor at the specified position."""
        quadrotor_id = p.loadURDF("quadrotor.urdf", position)
        
        # Change the color of the quadrotor
        for link_idx in range(p.getNumJoints(quadrotor_id) + 1):
            p.changeVisualShape(quadrotor_id, link_idx - 1, rgbaColor=color)
            
        return quadrotor_id
    
    def _add_target_drone(self):
        """Create the target drone (red sphere)."""
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
        return p.createMultiBody(baseMass=1, baseVisualShapeIndex=visual_shape_id, 
                               basePosition=self.target_position + self.offset)
    
    def _update_target_position(self):
        """Move the target freely within the space."""
        # Add small random variations to velocity
        self.target_velocity += np.random.uniform(-0.01, 0.01, size=3).astype(np.float32)
        
        # Normalize and scale velocity 
        self.target_velocity = self.target_velocity / (np.linalg.norm(self.target_velocity) + 1e-5) * self.target_speed
        
        # Update position
        self.target_position += self.target_velocity
        
        # Bounce off boundaries
        for i in range(3):
            if self.target_position[i] <= 0 or self.target_position[i] >= self.boundary[i]:
                self.target_velocity[i] *= -1
                self.target_position[i] = np.clip(self.target_position[i], 0, self.boundary[i])
        
        # Update target position in simulation
        p.resetBasePositionAndOrientation(self.target_drone, self.target_position + self.offset, [0, 0, 0, 1])
    
    def _get_neighbors(self, drone_index):
        """Get indices of neighboring drones within perception radius."""
        neighbors = []
        for i in range(self.num_drones):
            if i != drone_index:
                distance = np.linalg.norm(self.drone_positions[drone_index] - self.drone_positions[i])
                if distance < self.perception_radius:
                    neighbors.append(i)
        return neighbors
    
    def _separation_force(self, drone_index, neighbors):
        """Calculate separation force to avoid crowding neighbors."""
        if not neighbors:
            return np.zeros(3)
        
        force = np.zeros(3)
        for neighbor_idx in neighbors:
            diff = self.drone_positions[drone_index] - self.drone_positions[neighbor_idx]
            dist = np.linalg.norm(diff)
            if dist > 0:
                # Force is inversely proportional to distance
                force += diff / (dist * dist)
        
        return force
    
    def _alignment_force(self, drone_index, neighbors):
        """Calculate alignment force to steer towards average heading of neighbors."""
        if not neighbors:
            return np.zeros(3)
        
        avg_velocity = np.zeros(3)
        for neighbor_idx in neighbors:
            avg_velocity += self.drone_velocities[neighbor_idx]
        
        avg_velocity /= len(neighbors)
        return avg_velocity - self.drone_velocities[drone_index]
    
    def _cohesion_force(self, drone_index, neighbors):
        """Calculate cohesion force to move toward center mass of neighbors."""
        if not neighbors:
            return np.zeros(3)
        
        center_of_mass = np.zeros(3)
        for neighbor_idx in neighbors:
            center_of_mass += self.drone_positions[neighbor_idx]
        
        center_of_mass /= len(neighbors)
        return center_of_mass - self.drone_positions[drone_index]
    
    def _target_force(self, drone_index):
        """Calculate force to move toward target."""
        return self.target_position - self.drone_positions[drone_index]
    
    def _update_drone_velocities(self):
        """Update drone velocities based on boids rules."""
        new_velocities = np.copy(self.drone_velocities)
        
        for i in range(self.num_drones):
            neighbors = self._get_neighbors(i)
            
            # Calculate the four forces
            separation = self._separation_force(i, neighbors)
            alignment = self._alignment_force(i, neighbors)
            cohesion = self._cohesion_force(i, neighbors)
            target = self._target_force(i)
            
            # Normalize forces
            for force in [separation, alignment, cohesion, target]:
                norm = np.linalg.norm(force)
                if norm > 0:
                    force /= norm
            
            # Apply weighted forces
            new_velocities[i] += (
                self.separation_weight * separation +
                self.alignment_weight * alignment +
                self.cohesion_weight * cohesion +
                self.target_weight * target
            )
            
            # Limit speed
            speed = np.linalg.norm(new_velocities[i])
            if speed > self.max_speed:
                new_velocities[i] = (new_velocities[i] / speed) * self.max_speed
            elif speed < self.min_speed:
                new_velocities[i] = (new_velocities[i] / speed) * self.min_speed
        
        return new_velocities
    
    def _update_trails(self):
        """Update the trail lines for each drone."""
        # First remove old lines
        for drone_lines in self.line_ids:
            for line_id in drone_lines:
                p.removeUserDebugItem(line_id)
        
        # Clear line lists
        self.line_ids = [[] for _ in range(self.num_drones)]
        
        # Add latest positions to history
        for i in range(self.num_drones):
            self.position_history[i].append(self.drone_positions[i].copy())
        
        # Draw new lines
        for i in range(self.num_drones):
            positions = list(self.position_history[i])
            if len(positions) < 2:
                continue
            
            for j in range(len(positions) - 1):
                line_id = p.addUserDebugLine(
                    lineFromXYZ=positions[j] + self.offset,
                    lineToXYZ=positions[j + 1] + self.offset,
                    lineColorRGB=[0, 0.5 + 0.5 * (i / self.num_drones), 1],  # Match drone color
                    lineWidth=1.0,
                    lifeTime=0  # Persistent until removed
                )
                self.line_ids[i].append(line_id)
    
    def step(self, action):
        """Update environment for one step using Boids algorithm."""
        # Update target position
        self._update_target_position()
        
        # Calculate new velocities using Boids rules
        self.drone_velocities = self._update_drone_velocities()
        
        # Update positions
        self.drone_positions += self.drone_velocities
        
        # Ensure within boundaries
        for i in range(self.num_drones):
            for j in range(3):
                if self.drone_positions[i][j] < 0 or self.drone_positions[i][j] > self.boundary[j]:
                    # Bounce off boundary
                    self.drone_velocities[i][j] *= -0.5  # Dampen the bounce
                    self.drone_positions[i][j] = np.clip(self.drone_positions[i][j], 0, self.boundary[j])
        
        # Update trails
        self._update_trails()
        
        # Update drone positions in simulation
        for i in range(self.num_drones):
            _, orientation = p.getBasePositionAndOrientation(self.drones[i])
            p.resetBasePositionAndOrientation(self.drones[i], 
                                           self.drone_positions[i] + self.offset, 
                                           orientation)
        
        # Calculate reward as negative mean distance to target
        mean_distance = np.mean([np.linalg.norm(self.drone_positions[i] - self.target_position) 
                              for i in range(self.num_drones)])
        reward = -mean_distance
        
        return self._get_observation(), reward, False, {}
    
    def reset(self):
        """Reset the environment."""
        # Reset target
        self.target_position = np.array([5, 5, 5], dtype=np.float32)
        self.target_velocity = np.random.uniform(-self.target_speed, self.target_speed, size=3).astype(np.float32)
        
        # Reset drones
        self.drone_positions = np.random.rand(self.num_drones, 3) * self.boundary * 0.8 + self.boundary * 0.1
        self.drone_velocities = np.random.randn(self.num_drones, 3) * 0.01
        
        # Reset position history and clear trails
        for i in range(self.num_drones):
            self.position_history[i].clear()
            self.position_history[i].append(self.drone_positions[i].copy())
            
            for line_id in self.line_ids[i]:
                p.removeUserDebugItem(line_id)
        
        self.line_ids = [[] for _ in range(self.num_drones)]
        
        # Reset drone positions in simulation
        for i in range(self.num_drones):
            p.resetBasePositionAndOrientation(self.drones[i], 
                                           self.drone_positions[i] + self.offset, 
                                           [0, 0, 0, 1])
        
        # Reset target drone
        p.resetBasePositionAndOrientation(self.target_drone, 
                                       self.target_position + self.offset, 
                                       [0, 0, 0, 1])
        
        return self._get_observation()
    
    def _get_observation(self):
        """Return the observation."""
        return {
            'target_position': self.target_position,
            'drone_positions': self.drone_positions,
            'drone_velocities': self.drone_velocities
        }
    
    def close(self):
        """Clean up debug visualizations"""
        for drone_lines in self.line_ids:
            for line_id in drone_lines:
                p.removeUserDebugItem(line_id)

# Leader-Follower Algorithm Implementation
class UAVSwarmEnvLeaderFollower:
    def __init__(self, offset):
        self.offset = offset
        self.num_drones = 5  # 1 leader + 4 followers
        self.swarm_speed = 0.05
        self.boundary = np.array([10, 10, 10])
        
        # Leader parameters
        self.leader_index = 0
        self.leader_speed = 0.07
        self.leader_target_offset = 1.0  # Leader stays this distance from target
        
        # Follower parameters
        self.formation_radius = 2.0  # Radius of formation circle
        self.formation_height_offset = 0.5  # Height offset for followers
        
        # Define the formation offsets (circle formation)
        self.formation_offsets = []
        for i in range(1, self.num_drones):
            angle = 2 * np.pi * (i - 1) / (self.num_drones - 1)
            offset = np.array([
                self.formation_radius * np.cos(angle),
                self.formation_radius * np.sin(angle),
                -self.formation_height_offset
            ])
            self.formation_offsets.append(offset)
        
        # Path trail parameters
        self.trail_length = 50
        self.line_ids = [[] for _ in range(self.num_drones)]
        self.position_history = [deque(maxlen=self.trail_length) for _ in range(self.num_drones)]
        
        # Initialize target
        self.target_speed = 0.03
        self.target_position = np.array([5, 5, 5], dtype=np.float32)
        self.target_velocity = np.random.uniform(-self.target_speed, self.target_speed, size=3)
        self.target_drone = self._add_target_drone()
        
        # Initialize drones
        self.drones = []
        self.drone_positions = np.zeros((self.num_drones, 3))
        self.drone_velocities = np.zeros((self.num_drones, 3))
        
        # Initialize leader (yellow)
        self.drone_positions[self.leader_index] = self.target_position + np.array([0, 0, -self.leader_target_offset])
        leader_color = [1, 1, 0, 1]  # Yellow
        leader_id = self._create_quadrotor(self.drone_positions[self.leader_index] + self.offset, leader_color)
        self.drones.append(leader_id)
        self.position_history[self.leader_index].append(self.drone_positions[self.leader_index].copy())
        
        # Initialize followers (green)
        for i in range(1, self.num_drones):
            self.drone_positions[i] = self.drone_positions[self.leader_index] + self.formation_offsets[i-1]
            follower_color = [0, 1, 0, 1]  # Green
            follower_id = self._create_quadrotor(self.drone_positions[i] + self.offset, follower_color)
            self.drones.append(follower_id)
            self.position_history[i].append(self.drone_positions[i].copy())
    
    def _create_quadrotor(self, position, color):
        """Create a quadrotor at the specified position."""
        quadrotor_id = p.loadURDF("quadrotor.urdf", position)
        
        # Change the color of the quadrotor
        for link_idx in range(p.getNumJoints(quadrotor_id) + 1):
            p.changeVisualShape(quadrotor_id, link_idx - 1, rgbaColor=color)
            
        return quadrotor_id
    
    def _add_target_drone(self):
        """Create the target drone (red sphere)."""
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
        return p.createMultiBody(baseMass=1, baseVisualShapeIndex=visual_shape_id, 
                               basePosition=self.target_position + self.offset)
    
    def _update_target_position(self):
        """Move the target freely within the space."""
        # Add small random variations to velocity
        self.target_velocity += np.random.uniform(-0.01, 0.01, size=3)
        
        # Normalize and scale velocity 
        self.target_velocity = self.target_velocity / (np.linalg.norm(self.target_velocity) + 1e-5) * self.target_speed
        
        # Update position
        self.target_position += self.target_velocity
        
        # Bounce off boundaries
        for i in range(3):
            if self.target_position[i] <= 0 or self.target_position[i] >= self.boundary[i]:
                self.target_velocity[i] *= -1
                self.target_position[i] = np.clip(self.target_position[i], 0, self.boundary[i])
        
        # Update target position in simulation
        p.resetBasePositionAndOrientation(self.target_drone, self.target_position + self.offset, [0, 0, 0, 1])
    
    def _update_leader_position(self):
        """Update leader position to follow target with offset."""
        desired_position = self.target_position + np.array([0, 0, -self.leader_target_offset])
        direction = desired_position - self.drone_positions[self.leader_index]
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:
            # Move toward desired position
            self.drone_velocities[self.leader_index] = direction / distance * self.leader_speed
        else:
            # Close enough, slow down
            self.drone_velocities[self.leader_index] *= 0.9
        
        # Update position
        self.drone_positions[self.leader_index] += self.drone_velocities[self.leader_index]
        
        # Ensure within boundaries
        for j in range(3):
            if self.drone_positions[self.leader_index][j] < 0 or self.drone_positions[self.leader_index][j] > self.boundary[j]:
                self.drone_velocities[self.leader_index][j] *= -0.5
                self.drone_positions[self.leader_index][j] = np.clip(self.drone_positions[self.leader_index][j], 0, self.boundary[j])
    
    def _update_follower_positions(self):
        """Update follower positions to maintain formation."""
        leader_pos = self.drone_positions[self.leader_index]
        leader_vel = self.drone_velocities[self.leader_index]
        
        # Calculate leader orientation based on velocity
        if np.linalg.norm(leader_vel) > 0.01:
            forward = leader_vel / np.linalg.norm(leader_vel)
            # Assuming world up is [0, 0, 1]
            right = np.cross(forward, np.array([0, 0, 1]))
            if np.linalg.norm(right) > 0.01:
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)
            else:
                right = np.array([1, 0, 0])
                up = np.array([0, 1, 0])
        else:
            forward = np.array([1, 0, 0])
            right = np.array([0, 1, 0])
            up = np.array([0, 0, 1])
        
        # Rotation matrix from leader's orientation
        rot_matrix = np.column_stack((right, forward, up))
        
        # Update each follower
        for i in range(1, self.num_drones):
            # Calculate desired position in leader's frame
            offset_world = rot_matrix @ self.formation_offsets[i-1]
            desired_position = leader_pos + offset_world
            
            # Move toward desired position
            direction = desired_position - self.drone_positions[i]
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                self.drone_velocities[i] = direction / distance * self.swarm_speed
            else:
                # Close enough, match leader velocity
                self.drone_velocities[i] = leader_vel * 0.9
            
            # Update position
            self.drone_positions[i] += self.drone_velocities[i]
            
            # Ensure within boundaries
            for j in range(3):
                if self.drone_positions[i][j] < 0 or self.drone_positions[i][j] > self.boundary[j]:
                    self.drone_velocities[i][j] *= -0.5
                    self.drone_positions[i][j] = np.clip(self.drone_positions[i][j], 0, self.boundary[j])
    
    def _update_trails(self):
        """Update the trail lines for each drone."""
        # First remove old lines
        for drone_lines in self.line_ids:
            for line_id in drone_lines:
                p.removeUserDebugItem(line_id)
        
        # Clear line lists
        self.line_ids = [[] for _ in range(self.num_drones)]
        
        # Add latest positions to history
        for i in range(self.num_drones):
            self.position_history[i].append(self.drone_positions[i].copy())
        
        # Draw new lines
        for i in range(self.num_drones):
            positions = list(self.position_history[i])
            if len(positions) < 2:
                continue
            
            # Set color based on drone type
            if i == self.leader_index:
                color = [1, 1, 0]  # Yellow for leader
            else:
                color = [0, 1, 0]  # Green for followers
            
            for j in range(len(positions) - 1):
                line_id = p.addUserDebugLine(
                    lineFromXYZ=positions[j] + self.offset,
                    lineToXYZ=positions[j + 1] + self.offset,
                    lineColorRGB=color,
                    lineWidth=1.0,
                    lifeTime=0  # Persistent until removed
                )
                self.line_ids[i].append(line_id)
    
    def step(self, action):
        """Update environment for one step using Leader-Follower algorithm."""
        # Update target position
        self._update_target_position()
        
        # Update leader position
        self._update_leader_position()
        
        # Update follower positions
        self._update_follower_positions()
        
        # Update trails
        self._update_trails()
        
        # Update drone positions in simulation
        for i in range(self.num_drones):
            _, orientation = p.getBasePositionAndOrientation(self.drones[i])
            p.resetBasePositionAndOrientation(self.drones[i], 
                                           self.drone_positions[i] + self.offset, 
                                           orientation)
        
        # Calculate reward based on formation accuracy
        formation_error = 0
        leader_pos = self.drone_positions[self.leader_index]
        
        for i in range(1, self.num_drones):
            # Calculate ideal position based on leader
            ideal_pos = leader_pos + self.formation_offsets[i-1]
            # Calculate error
            error = np.linalg.norm(self.drone_positions[i] - ideal_pos)
            formation_error += error
        
        formation_error /= (self.num_drones - 1)
        reward = -formation_error
        
        return self._get_observation(), reward, False, {}
    
    def reset(self):
        """Reset the environment."""
        # Reset target
        self.target_position = np.array([5, 5, 5], dtype=np.float32)
        self.target_velocity = np.random.uniform(-self.target_speed, self.target_speed, size=3)
        
        # Reset leader
        self.drone_positions[self.leader_index] = self.target_position + np.array([0, 0, -self.leader_target_offset])
        self.drone_velocities[self.leader_index] = np.zeros(3)
        
        # Reset followers
        for i in range(1, self.num_drones):
            self.drone_positions[i] = self.drone_positions[self.leader_index] + self.formation_offsets[i-1]
            self.drone_velocities[i] = np.zeros(3)
        
        # Reset position history and clear trails
        for i in range(self.num_drones):
            self.position_history[i].clear()
            self.position_history[i].append(self.drone_positions[i].copy())
            
            for line_id in self.line_ids[i]:
                p.removeUserDebugItem(line_id)
        
        self.line_ids = [[] for _ in range(self.num_drones)]
        
        # Reset drone positions in simulation
        for i in range(self.num_drones):
            p.resetBasePositionAndOrientation(self.drones[i], 
                                           self.drone_positions[i] + self.offset, 
                                           [0, 0, 0, 1])
        
        # Reset target drone
        p.resetBasePositionAndOrientation(self.target_drone, 
                                       self.target_position + self.offset, 
                                       [0, 0, 0, 1])
        
        return self._get_observation()
    
    def _get_observation(self):
        """Return the observation."""
        return {
            'target_position': self.target_position,
            'leader_position': self.drone_positions[0],
            'follower_positions': self.drone_positions[1:],
            'formation_errors': [np.linalg.norm(self.drone_positions[i] - 
                                           (self.drone_positions[0] + self.formation_offsets[i-1])) 
                             for i in range(1, self.num_drones)]
        }
    
    def close(self):
        """Clean up debug visualizations"""
        for drone_lines in self.line_ids:
            for line_id in drone_lines:
                p.removeUserDebugItem(line_id)


if __name__ == "__main__":
    env = UAVSwarmEnvMulti()
    env.run_all()