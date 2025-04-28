import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from collections import deque
import matplotlib.pyplot as plt


class UAVSwarmEnvABC(gym.Env):
    def __init__(self):
        super(UAVSwarmEnvABC, self).__init__()

        # ABC algorithm parameters - reduced to 5 quadrotors
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
        
        # Initialize PyBullet simulation
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # No gravity for UAVs
        self.plane = p.loadURDF("plane.urdf")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
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
            drone_id = self._create_quadrotor(self.drone_positions[i], color)
            self.drones.append(drone_id)
            
            # Initialize position history for trail
            self.position_history[i].append(self.drone_positions[i].copy())
            
        # NEW: Metrics tracking
        self.tracking_accuracy_history = []  # Average distance to target
        self.swarm_stability_history = []    # Standard deviation of inter-drone distances
        self.convergence_threshold = 1.5     # Threshold for convergence (distance to target)
        self.converged = False
        self.convergence_time = None
        self.simulation_steps = 0

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
        return p.createMultiBody(baseMass=1, baseVisualShapeIndex=visual_shape_id, basePosition=self.target_position)
    
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
        p.resetBasePositionAndOrientation(self.target_drone, self.target_position, [0, 0, 0, 1])
    
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
                    lineFromXYZ=positions[j],
                    lineToXYZ=positions[j + 1],
                    lineColorRGB=self.drone_colors[i][:3],  # Use drone color for line
                    lineWidth=1.0,
                    lifeTime=0  # Persistent until removed
                )
                self.line_ids[i].append(line_id)
    
    # NEW: Calculate and update metrics
    def _update_metrics(self):
        """Calculate and update the tracking metrics."""
        # Increment simulation step counter
        self.simulation_steps += 1
        
        # 1. Tracking Accuracy - Average distance to target
        distances_to_target = [np.linalg.norm(self.drone_positions[i] - self.target_position) 
                             for i in range(self.num_drones)]
        avg_distance = np.mean(distances_to_target)
        self.tracking_accuracy_history.append(avg_distance)
        
        # 2. Swarm Stability - Standard deviation of inter-drone distances
        inter_drone_distances = []
        for i in range(self.num_drones):
            for j in range(i+1, self.num_drones):
                distance = np.linalg.norm(self.drone_positions[i] - self.drone_positions[j])
                inter_drone_distances.append(distance)
        
        stability = np.std(inter_drone_distances) if inter_drone_distances else 0
        self.swarm_stability_history.append(stability)
        
        # 3. Check convergence (if not already converged)
        if not self.converged and avg_distance < self.convergence_threshold:
            self.converged = True
            self.convergence_time = self.simulation_steps
    
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
        
        # 6. Update metrics
        self._update_metrics()
        
        # 7. Update drone positions in simulation
        for i in range(self.num_drones):
            # Get current orientation
            _, orientation = p.getBasePositionAndOrientation(self.drones[i])
            
            # Calculate desired orientation (basic form - point in direction of movement)
            if np.linalg.norm(self.drone_velocities[i]) > 0.01:
                direction = self.drone_velocities[i] / np.linalg.norm(self.drone_velocities[i])
                # This is a simplistic approach - in a real implementation you would want
                # proper orientation calculations based on quadrotor dynamics
                
            # Reset position (and orientation if needed)
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i], orientation)
        
        # 8. Step simulation
        p.stepSimulation()
        
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
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i], [0, 0, 0, 1])
        
        # Reset metrics
        self.tracking_accuracy_history = []
        self.swarm_stability_history = []
        self.converged = False
        self.convergence_time = None
        self.simulation_steps = 0
        
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
        """Close the PyBullet simulation."""
        # Clean up debug visualizations
        for drone_lines in self.line_ids:
            for line_id in drone_lines:
                p.removeUserDebugItem(line_id)
        p.disconnect()
    
    # NEW: Generate and save metrics plots
    def generate_metrics_plots(self, save_dir="ABC_metrics_png"):
        """Generate and save metrics plots."""
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 1. Tracking Accuracy Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.tracking_accuracy_history)
        plt.title('Tracking Accuracy: Average Distance to Target vs Time')
        plt.xlabel('Simulation Step')
        plt.ylabel('Average Distance')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'tracking_accuracy.png'))
        plt.close()
        
        # 2. Swarm Stability Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.swarm_stability_history)
        plt.title('Swarm Stability: Std Dev of Inter-Drone Distances vs Time')
        plt.xlabel('Simulation Step')
        plt.ylabel('Standard Deviation')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'swarm_stability.png'))
        plt.close()
        
        # 3. Convergence Time Bar Chart
        plt.figure(figsize=(8, 6))
        if self.convergence_time is not None:
            plt.bar(['Convergence Time'], [self.convergence_time], color='green')
            plt.title(f'Convergence Time: {self.convergence_time} Steps')
        else:
            plt.bar(['Convergence Time'], [self.simulation_steps], color='red')
            plt.title(f'Did Not Converge within {self.simulation_steps} Steps')
        plt.ylabel('Number of Steps')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(save_dir, 'convergence_time.png'))
        plt.close()
        
        print(f"Plots saved to {save_dir}/ directory")


def display_info():
    """Display simulation information in PyBullet GUI."""
    p.addUserDebugText(
        "Artificial Bee Colony (ABC) Algorithm",
        [0, 0, 3],
        textColorRGB=[1, 1, 1],
        textSize=1.5
    )
    p.addUserDebugText(
        "Blue: Employed Bees | Green: Onlooker Bees | Red: Target",
        [0, 0, 2.5],
        textColorRGB=[1, 1, 1],
        textSize=1.2
    )


if __name__ == "__main__":
    env = UAVSwarmEnvABC()
    obs = env.reset()
    display_info()
    
    # Main simulation loop
    try:
        max_steps = 500  # Set a maximum number of steps
        for step in range(max_steps):
            action = np.zeros((env.num_drones, 3))  # No external actions needed
            obs, reward, done, info = env.step(action)
            
            # Find best food source
            best_idx = np.argmax(env.food_qualities)
            best_quality = env.food_qualities[best_idx]
            best_position = env.food_positions[best_idx]
            distance = np.linalg.norm(best_position - env.target_position)
            
            # Display current metrics for this step
            avg_distance = env.tracking_accuracy_history[-1]
            stability = env.swarm_stability_history[-1]
            
            print(f"Step: {step} | Avg Distance: {avg_distance:.2f} | Stability: {stability:.2f} | Best solution quality: {best_quality:.2f}")
            
            # Check if swarm has converged
            if env.converged and env.convergence_time == step:
                print(f"Swarm converged at step {step}!")
            
            time.sleep(0.02)  # Control simulation speed
            
            # Optional: End simulation early if converged for some time
            if env.converged and step > env.convergence_time + 100:
                print("Swarm has been stable for a while. Ending simulation.")
                break
                
    except KeyboardInterrupt:
        print("Simulation ended by user")
    
    # Generate and save metrics plots
    env.generate_metrics_plots()
    
    env.close()