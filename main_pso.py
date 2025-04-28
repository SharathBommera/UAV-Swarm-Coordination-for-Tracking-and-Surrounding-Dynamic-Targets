import gym
import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

class UAVSwarmEnvPSO(gym.Env):
    def __init__(self):
        super(UAVSwarmEnvPSO, self).__init__()

        self.num_drones = 5
        self.swarm_speed = 0.05
        self.target_speed = 0.03
        self.boundary = np.array([10, 10, 10])

        # Metrics logging
        self.tracking_accuracy = []  # Average distance to target
        self.swarm_stability = []    # Std deviation of inter-drone distances
        self.simulation_steps = 0
        self.converged = False
        self.convergence_time = None
        self.convergence_threshold = 2.0  # Distance threshold for convergence

        self.path_colors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]
        ]
        self.previous_positions = []

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")

        self.target_position = np.random.rand(3) * 5 + np.array([3, 3, 3])
        self.target_velocity = np.random.randn(3) * self.target_speed
        self.target_drone = self._add_target_drone()

        self.drones = []
        self.drone_positions = np.random.rand(self.num_drones, 3) * 5
        self.drone_velocities = np.random.randn(self.num_drones, 3) * self.swarm_speed
        self.personal_best_positions = np.copy(self.drone_positions)
        self.global_best_position = np.mean(self.drone_positions, axis=0)

        for i in range(self.num_drones):
            drone_id = p.loadURDF("quadrotor.urdf", self.drone_positions[i])
            self.drones.append(drone_id)
            self.previous_positions.append(self.drone_positions[i].copy())

    def _add_target_drone(self):
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=self.target_position)

    def _update_target_position(self):
        self.target_position += self.target_velocity
        for i in range(3):
            if self.target_position[i] < 0 or self.target_position[i] > self.boundary[i]:
                self.target_velocity[i] *= -1
        self.target_velocity += np.random.uniform(-0.01, 0.01, size=3)
        self.target_velocity = np.clip(self.target_velocity, -self.target_speed, self.target_speed)
        p.resetBasePositionAndOrientation(self.target_drone, self.target_position, [0, 0, 0, 1])

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

        self.global_best_position = np.mean(self.personal_best_positions, axis=0)
        return new_velocities

    def step(self, action):
        self._update_target_position()
        self.drone_velocities = self._pso_behavior()
        self.drone_positions += self.drone_velocities

        for i in range(self.num_drones):
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i], [0, 0, 0, 1])
            p.addUserDebugLine(self.previous_positions[i], self.drone_positions[i], self.path_colors[i], 1)
            self.previous_positions[i] = self.drone_positions[i].copy()

        p.stepSimulation()
        
        # Update metrics
        self.simulation_steps += 1
        self._update_metrics()
        
        return self._get_observation(), self._compute_reward(), False, {}

    def _update_metrics(self):
        # Calculate tracking accuracy (average distance to target)
        distances_to_target = [np.linalg.norm(pos - self.target_position) for pos in self.drone_positions]
        avg_distance = np.mean(distances_to_target)
        self.tracking_accuracy.append(avg_distance)
        
        # Calculate swarm stability (std dev of inter-drone distances)
        if self.num_drones > 1:
            inter_drone_distances = pdist(self.drone_positions)
            stability = np.std(inter_drone_distances)
            self.swarm_stability.append(stability)
        else:
            self.swarm_stability.append(0)
        
        # Check for convergence
        if not self.converged and avg_distance < self.convergence_threshold:
            consecutive_converged = 10  # Require 10 consecutive steps below threshold
            if len(self.tracking_accuracy) >= consecutive_converged:
                if all(d < self.convergence_threshold for d in self.tracking_accuracy[-consecutive_converged:]):
                    self.converged = True
                    self.convergence_time = self.simulation_steps
                    print(f"Convergence achieved at step {self.convergence_time}")

    def reset(self):
        self.target_position = np.random.rand(3) * 5 + np.array([3, 3, 3])
        self.target_velocity = np.random.randn(3) * self.target_speed
        self.drone_positions = np.random.rand(self.num_drones, 3) * 5
        self.drone_velocities = np.random.randn(self.num_drones, 3) * self.swarm_speed
        self.personal_best_positions = np.copy(self.drone_positions)
        self.global_best_position = np.mean(self.drone_positions, axis=0)
        self.previous_positions = [pos.copy() for pos in self.drone_positions]
        
        # Reset metrics
        self.tracking_accuracy = []
        self.swarm_stability = []
        self.simulation_steps = 0
        self.converged = False
        self.convergence_time = None

        for i in range(self.num_drones):
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i], [0, 0, 0, 1])
        return self._get_observation()

    def _get_observation(self):
        return np.hstack([self.target_position, self.drone_positions.flatten()])

    def _compute_reward(self):
        distance_to_target = np.mean([np.linalg.norm(self.drone_positions[i] - self.target_position) for i in range(self.num_drones)])
        return -distance_to_target

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
        
    def generate_metrics_plots(self, max_steps=None):
        """Generate and save plots for all metrics collected during simulation."""
        # Create output directory if it doesn't exist
        output_dir = "pso_metrics_png"
        os.makedirs(output_dir, exist_ok=True)
        
        # If max_steps is provided, limit the data points
        if max_steps and max_steps < len(self.tracking_accuracy):
            tracking_data = self.tracking_accuracy[:max_steps]
            stability_data = self.swarm_stability[:max_steps]
            steps = range(1, max_steps + 1)
        else:
            tracking_data = self.tracking_accuracy
            stability_data = self.swarm_stability
            steps = range(1, len(tracking_data) + 1)
        
        # Plot 1: Tracking Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(steps, tracking_data)
        plt.title('Tracking Accuracy: Average Distance to Target vs Time')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Average Distance to Target')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'tracking_accuracy.png'), dpi=300)
        plt.close()
        
        # Plot 2: Swarm Stability
        plt.figure(figsize=(10, 6))
        plt.plot(steps, stability_data)
        plt.title('Swarm Stability: Std Deviation of Inter-drone Distances vs Time')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Std Deviation of Inter-drone Distances')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'swarm_stability.png'), dpi=300)
        plt.close()
        
        # Plot 3: Convergence Time
        plt.figure(figsize=(8, 6))
        if self.convergence_time:
            plt.bar(['PSO Algorithm'], [self.convergence_time], color='blue')
            plt.title(f'Convergence Time: {self.convergence_time} Steps')
        else:
            plt.bar(['PSO Algorithm'], [len(self.tracking_accuracy)], color='red')
            plt.title('Convergence Time: Did Not Converge')
        plt.ylabel('Number of Steps to Converge')
        plt.savefig(os.path.join(output_dir, 'convergence_time.png'), dpi=300)
        plt.close()
        
        print(f"Metrics plots saved to '{output_dir}' directory")

if __name__ == "__main__":
    env = UAVSwarmEnvPSO()
    obs = env.reset()
    
    # Maximum number of simulation steps
    max_steps = 500
    
    try:
        for step in range(max_steps):
            action = np.zeros((env.num_drones, 3))
            obs, reward, done, info = env.step(action)
            print(f"Step {step+1}/{max_steps}, Reward: {reward:.2f}, Avg Distance: {env.tracking_accuracy[-1]:.2f}")
            time.sleep(0.01)  # Reduced sleep time for faster simulation
            
            # Optionally break early if converged for a while
            if env.converged and step > env.convergence_time + 50:
                print("Breaking early as convergence achieved")
                break
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        # Generate and save metrics plots
        env.generate_metrics_plots()
        env.close()