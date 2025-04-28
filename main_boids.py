import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

class UAVSwarmEnvBoids(gym.Env):
    def __init__(self):
        super(UAVSwarmEnvBoids, self).__init__()

        self.num_drones = 5
        self.swarm_speed = 0.05
        self.target_speed = self.swarm_speed
        self.boundary = np.array([10, 10, 10], dtype=np.float32)
        self.perception_radius = 3.0
        
        # Metrics tracking
        self.tracking_accuracy_history = []
        self.swarm_stability_history = []
        self.formation_threshold = 1.5  # Distance threshold for considering the swarm in formation
        self.converged = False
        self.convergence_time = None
        self.step_count = 0

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")

        self.target_position = np.array([5, 5, 5], dtype=np.float32)
        self.target_velocity = np.random.uniform(-self.target_speed, self.target_speed, size=3).astype(np.float32)
        self.target_drone = self._add_target_drone()

        self.drones = []
        self.drone_positions = np.random.rand(self.num_drones, 3) * 10
        self.drone_velocities = np.zeros((self.num_drones, 3), dtype=np.float32)

        for i in range(self.num_drones):
            drone_id = p.loadURDF("quadrotor.urdf", self.drone_positions[i])
            self.drones.append(drone_id)

        # Initialize previous positions for path drawing
        self.previous_positions = self.drone_positions.copy()
        self.path_colors = [
            [1, 0, 0],    # red
            [0, 1, 0],    # green
            [0, 0, 1],    # blue
            [1, 1, 0],    # yellow
            [1, 0, 1],    # magenta
        ]

    def _add_target_drone(self):
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
        return p.createMultiBody(baseMass=1, baseVisualShapeIndex=visual_shape_id, basePosition=self.target_position)

    def _update_target_position(self):
        self.target_velocity += np.random.uniform(-0.02, 0.02, size=3).astype(np.float32)
        self.target_velocity = self.target_velocity / (np.linalg.norm(self.target_velocity) + 1e-5) * self.target_speed
        self.target_position += self.target_velocity
        self.target_position = np.clip(self.target_position, 0, self.boundary)
        p.resetBasePositionAndOrientation(self.target_drone, self.target_position, [0, 0, 0, 1])

    def _boids_behavior(self):
        separation_weight = 1.5
        alignment_weight = 1.0
        cohesion_weight = 1.0
        target_attraction_weight = 2.0

        new_velocities = np.zeros_like(self.drone_velocities)

        for i in range(self.num_drones):
            separation = np.zeros(3)
            alignment = np.zeros(3)
            cohesion = np.zeros(3)
            neighbor_count = 0

            for j in range(self.num_drones):
                if i == j:
                    continue
                distance = np.linalg.norm(self.drone_positions[i] - self.drone_positions[j])
                if distance < self.perception_radius:
                    separation += (self.drone_positions[i] - self.drone_positions[j]) / (distance + 1e-5)
                    alignment += self.drone_velocities[j]
                    cohesion += self.drone_positions[j]
                    neighbor_count += 1

            if neighbor_count > 0:
                alignment /= neighbor_count
                cohesion = (cohesion / neighbor_count) - self.drone_positions[i]

            to_target = self.target_position - self.drone_positions[i]

            new_velocities[i] = (separation_weight * separation +
                                 alignment_weight * alignment +
                                 cohesion_weight * cohesion +
                                 target_attraction_weight * to_target)

            new_velocities[i] = new_velocities[i] / (np.linalg.norm(new_velocities[i]) + 1e-5) * self.swarm_speed

        return new_velocities

    def _calculate_metrics(self):
        # 1. Tracking Accuracy - average distance to target
        distances_to_target = np.array([
            np.linalg.norm(self.drone_positions[i] - self.target_position)
            for i in range(self.num_drones)
        ])
        avg_distance_to_target = np.mean(distances_to_target)
        self.tracking_accuracy_history.append(avg_distance_to_target)
        
        # 2. Swarm Stability - standard deviation of inter-drone distances
        if self.num_drones > 1:
            inter_drone_distances = pdist(self.drone_positions)
            stability = np.std(inter_drone_distances)
            self.swarm_stability_history.append(stability)
        else:
            self.swarm_stability_history.append(0)
        
        # 3. Check for convergence (if not already converged)
        if not self.converged and avg_distance_to_target < self.formation_threshold:
            self.converged = True
            self.convergence_time = self.step_count
            print(f"Swarm converged at step {self.convergence_time}")

    def step(self, action):
        self.step_count += 1
        self._update_target_position()
        self.drone_velocities = self._boids_behavior()
        self.drone_positions += self.drone_velocities

        # Draw trails before updating
        for i in range(self.num_drones):
            start = self.previous_positions[i]
            end = self.drone_positions[i]
            color = self.path_colors[i % len(self.path_colors)]
            p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=1.5, lifeTime=30)

        self.previous_positions = self.drone_positions.copy()

        for i in range(self.num_drones):
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i], [0, 0, 0, 1])

        # Calculate metrics
        self._calculate_metrics()
        
        p.stepSimulation()
        return self._get_observation(), self._compute_reward(), False, {}

    def reset(self):
        self.step_count = 0
        self.target_position = np.array([5, 5, 5], dtype=np.float32)
        self.target_velocity = np.random.uniform(-self.target_speed, self.target_speed, size=3).astype(np.float32)
        self.drone_positions = np.random.rand(self.num_drones, 3) * 10
        self.drone_velocities = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.previous_positions = self.drone_positions.copy()
        
        # Reset metrics
        self.tracking_accuracy_history = []
        self.swarm_stability_history = []
        self.converged = False
        self.convergence_time = None

        for i in range(self.num_drones):
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i], [0, 0, 0, 1])

        return self._get_observation()

    def _get_observation(self):
        return np.hstack([self.target_position, self.drone_positions.flatten()])

    def _compute_reward(self):
        distance_to_target = np.mean([
            np.linalg.norm(self.drone_positions[i] - self.target_position)
            for i in range(self.num_drones)
        ])
        return -distance_to_target

    def plot_metrics(self, output_dir="boids_metrics_png"):
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
            
        # 1. Plot tracking accuracy over time
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.tracking_accuracy_history)), self.tracking_accuracy_history)
        plt.title('Tracking Accuracy Over Time')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Average Distance to Target')
        plt.grid(True)
        plt.savefig(f"{output_dir}/tracking_accuracy.png", bbox_inches='tight')
        plt.close()
        
        # 2. Plot swarm stability over time
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.swarm_stability_history)), self.swarm_stability_history)
        plt.title('Swarm Stability Over Time')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Std Dev of Inter-Drone Distances')
        plt.grid(True)
        plt.savefig(f"{output_dir}/swarm_stability.png", bbox_inches='tight')
        plt.close()
        
        # 3. Plot convergence time as a bar chart
        plt.figure(figsize=(8, 6))
        if self.convergence_time is not None:
            plt.bar(['Convergence Time'], [self.convergence_time], color='blue')
        else:
            plt.bar(['Convergence Time'], [len(self.tracking_accuracy_history)], color='red')
            plt.title('Convergence Time (Did Not Converge)')
        plt.title('Convergence Time')
        plt.ylabel('Number of Steps')
        plt.grid(axis='y')
        plt.savefig(f"{output_dir}/convergence_time.png", bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {output_dir}/")

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    env = UAVSwarmEnvBoids()
    obs = env.reset()

    # Run simulation for a fixed number of steps
    max_steps = 500
    for step in range(max_steps):
        action = np.zeros((env.num_drones, 3))
        obs, reward, done, info = env.step(action)
        
        if step % 50 == 0:
            print(f"Step {step}/{max_steps}, Reward: {reward:.2f}")
        
        time.sleep(0.01)  # Speed up simulation
    
    # Generate and save metrics plots
    env.plot_metrics()
    
    # Keep the window open for a bit longer
    print("Simulation completed. Generated metric plots.")
    time.sleep(3)
    
    env.close()