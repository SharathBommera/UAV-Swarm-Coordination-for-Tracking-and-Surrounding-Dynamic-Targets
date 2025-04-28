import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class UAVSwarmEnv(gym.Env):
    def __init__(self):
        super(UAVSwarmEnv, self).__init__()

        self.num_drones = 5
        self.leader_index = 0
        self.formation_radius = 3.0
        self.swarm_speed = 0.05
        self.target_speed = 0.03
        self.boundary = np.array([10, 10, 10])  # Target boundary restriction
        
        # Metrics tracking
        self.tracking_accuracy = []  # Average distance between drones and target
        self.swarm_stability = []    # Standard deviation of distances between leader and followers
        self.convergence_time = None # Steps taken to stabilize formation
        self.convergence_threshold = 0.5  # Threshold for considering formation stabilized
        self.formation_stabilized = False
        self.simulation_steps = 0
        
        # Connect to PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")

        # Target drone (bright red sphere)
        self.target_position = np.random.rand(3) * 5 + np.array([5, 5, 5])  # Initial position within 10x10x10 region
        self.target_velocity = np.random.randn(3) * self.target_speed
        self.target_drone = self._add_target_drone()

        # Drone initialization
        self.drones = []
        self.drone_positions = np.random.rand(self.num_drones, 3) * 10
        self.drone_velocities = np.zeros((self.num_drones, 3))

        for i in range(self.num_drones):
            if i == self.leader_index:
                drone_id = self._create_colored_drone(self.drone_positions[i], color=[1, 1, 0, 1])  # Yellow leader
            else:
                drone_id = p.loadURDF("quadrotor.urdf", self.drone_positions[i])
            self.drones.append(drone_id)

        self.follower_offsets = self._generate_circle_offsets()

        # For path trail drawing
        self.previous_positions = self.drone_positions.copy()
        self.path_colors = [
            [1, 0, 0],    # red
            [0, 1, 0],    # green
            [0, 0, 1],    # blue
            [1, 1, 0],    # yellow
            [1, 0, 1],    # magenta
        ]

    def _generate_circle_offsets(self):
        angles = np.linspace(0, 2 * np.pi, self.num_drones - 1, endpoint=False)
        offsets = np.array([[self.formation_radius * np.cos(a),
                             self.formation_radius * np.sin(a),
                             0] for a in angles])
        return offsets

    def _add_target_drone(self):
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[1, 0, 0, 1])  # Bigger and brighter red
        return p.createMultiBody(baseMass=1,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=self.target_position)

    def _create_colored_drone(self, position, color):
        visual_shape_id = p.createVisualShape(p.GEOM_BOX,
                                              halfExtents=[0.15, 0.15, 0.05],
                                              rgbaColor=color)
        collision_shape_id = p.createCollisionShape(p.GEOM_BOX,
                                                    halfExtents=[0.15, 0.15, 0.05])
        return p.createMultiBody(baseMass=1,
                                 baseCollisionShapeIndex=collision_shape_id,
                                 baseVisualShapeIndex=visual_shape_id,
                                 basePosition=position)

    def _update_target_position(self):
        self.target_position += self.target_velocity

        # Ensure target stays within the 10x10x10 region
        for i in range(3):
            if self.target_position[i] < 0:
                self.target_position[i] = 0
                self.target_velocity[i] *= -1
            elif self.target_position[i] > self.boundary[i]:
                self.target_position[i] = self.boundary[i]
                self.target_velocity[i] *= -1

        self.target_velocity += np.random.uniform(-0.01, 0.01, size=3)
        self.target_velocity = np.clip(self.target_velocity, -self.target_speed, self.target_speed)

        p.resetBasePositionAndOrientation(self.target_drone, self.target_position, [0, 0, 0, 1])
        
    def _calculate_metrics(self):
        # Calculate tracking accuracy (average distance to target)
        distances_to_target = np.array([np.linalg.norm(self.drone_positions[i] - self.target_position) 
                                       for i in range(self.num_drones)])
        avg_distance_to_target = np.mean(distances_to_target)
        self.tracking_accuracy.append(avg_distance_to_target)
        
        # Calculate swarm stability (std dev of distances between leader and followers)
        leader_pos = self.drone_positions[self.leader_index]
        follower_distances = np.array([np.linalg.norm(self.drone_positions[i] - leader_pos) 
                                      for i in range(self.num_drones) if i != self.leader_index])
        stability = np.std(follower_distances)
        self.swarm_stability.append(stability)
        
        # Check for convergence if not already converged
        if not self.formation_stabilized:
            # Calculate ideal positions for followers
            ideal_positions = np.array([self.drone_positions[self.leader_index] + offset 
                                       for offset in self.follower_offsets])
            
            # Calculate actual distances from ideal positions
            follower_idx = 0
            formation_errors = []
            for i in range(self.num_drones):
                if i != self.leader_index:
                    error = np.linalg.norm(self.drone_positions[i] - ideal_positions[follower_idx])
                    formation_errors.append(error)
                    follower_idx += 1
            
            max_error = max(formation_errors)
            if max_error < self.convergence_threshold:
                self.formation_stabilized = True
                self.convergence_time = self.simulation_steps

    def step(self, action):
        self.simulation_steps += 1
        self._update_target_position()

        leader_desired_position = self.target_position + np.array([0, 0, 1])
        leader_dir = leader_desired_position - self.drone_positions[self.leader_index]
        leader_dist = np.linalg.norm(leader_dir)

        if leader_dist > 0.05:
            leader_dir = leader_dir / leader_dist
            self.drone_velocities[self.leader_index] = leader_dir * self.swarm_speed
        else:
            self.drone_velocities[self.leader_index] = np.zeros(3)

        self.drone_positions[self.leader_index] += self.drone_velocities[self.leader_index]
        p.resetBasePositionAndOrientation(self.drones[self.leader_index],
                                          self.drone_positions[self.leader_index],
                                          [0, 0, 0, 1])

        offset_idx = 0
        for i in range(self.num_drones):
            if i == self.leader_index:
                continue

            desired_position = self.drone_positions[self.leader_index] + self.follower_offsets[offset_idx]
            direction = desired_position - self.drone_positions[i]
            distance = np.linalg.norm(direction)

            if distance > 0.05:
                direction = direction / distance
                self.drone_velocities[i] = direction * self.swarm_speed
            else:
                self.drone_velocities[i] = np.zeros(3)

            self.drone_positions[i] += self.drone_velocities[i]
            p.resetBasePositionAndOrientation(self.drones[i],
                                              self.drone_positions[i],
                                              [0, 0, 0, 1])
            offset_idx += 1

        # Calculate and store metrics
        self._calculate_metrics()

        # Draw path trails
        for i in range(self.num_drones):
            start = self.previous_positions[i]
            end = self.drone_positions[i]
            color = self.path_colors[i % len(self.path_colors)]
            p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=1.5, lifeTime=30)

        self.previous_positions = self.drone_positions.copy()

        p.stepSimulation()
        return self._get_observation(), self._compute_reward(), False, {}

    def reset(self):
        self.target_position = np.random.rand(3) * 5 + np.array([5, 5, 5])
        self.target_velocity = np.random.randn(3) * self.target_speed

        self.drone_positions = np.random.rand(self.num_drones, 3) * 10
        self.drone_velocities = np.zeros((self.num_drones, 3))
        self.previous_positions = self.drone_positions.copy()

        for i in range(self.num_drones):
            p.resetBasePositionAndOrientation(self.drones[i], self.drone_positions[i], [0, 0, 0, 1])
            
        # Reset metrics tracking
        self.tracking_accuracy = []
        self.swarm_stability = []
        self.convergence_time = None
        self.formation_stabilized = False
        self.simulation_steps = 0

        return self._get_observation()

    def _get_observation(self):
        return np.hstack([self.target_position, self.drone_positions.flatten()])

    def _compute_reward(self):
        penalty = 0
        offset_idx = 0
        for i in range(self.num_drones):
            if i == self.leader_index:
                continue
            desired = self.drone_positions[self.leader_index] + self.follower_offsets[offset_idx]
            penalty += np.linalg.norm(self.drone_positions[i] - desired)
            offset_idx += 1
        return -penalty / (self.num_drones - 1)
        
    def generate_metric_plots(self, max_steps=None):
        """Generate and save metric plots as PNG images"""
        # Create directory if it doesn't exist
        output_dir = "leader_follower_metrics_png"
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare x-axis (time steps)
        if max_steps is None:
            max_steps = len(self.tracking_accuracy)
        steps = np.arange(max_steps)
        
        # Plot 1: Tracking Accuracy
        fig1 = plt.figure(figsize=(10, 6))
        plt.plot(steps, self.tracking_accuracy[:max_steps], 'b-', linewidth=2)
        plt.title('Tracking Accuracy: Average Distance to Target')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Average Distance (units)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'tracking_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Plot 2: Swarm Stability
        fig2 = plt.figure(figsize=(10, 6))
        plt.plot(steps, self.swarm_stability[:max_steps], 'g-', linewidth=2)
        plt.title('Swarm Stability: StdDev of Distances Between Leader and Followers')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Standard Deviation (units)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'swarm_stability.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Plot 3: Convergence Time
        fig3 = plt.figure(figsize=(8, 6))
        conv_time = self.convergence_time if self.convergence_time is not None else max_steps
        plt.bar(['Formation Convergence'], [conv_time], color='orange', width=0.4)
        plt.title('Convergence Time: Steps to Stable Formation')
        plt.ylabel('Number of Steps')
        plt.grid(axis='y')
        plt.savefig(os.path.join(output_dir, 'convergence_time.png'), dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        print(f"Metric plots saved to '{output_dir}' directory")

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = UAVSwarmEnv()
    obs = env.reset()
    
    # Maximum number of simulation steps
    max_steps = 500
    
    print("Starting simulation with metric tracking...")
    for step in range(max_steps):
        action = np.zeros((env.num_drones, 3))
        obs, reward, done, info = env.step(action)
        
        # Print some metrics every 20 steps
        if step % 20 == 0:
            print(f"Step {step}/{max_steps}")
            print(f"  Tracking Accuracy: {env.tracking_accuracy[-1]:.2f}")
            print(f"  Swarm Stability: {env.swarm_stability[-1]:.2f}")
            if env.formation_stabilized:
                print(f"  Formation Stabilized at step {env.convergence_time}")
            else:
                print("  Formation not yet stabilized")
        
        time.sleep(0.01)  # Speed up simulation a bit
    
    print("\nSimulation complete. Generating metric plots...")
    env.generate_metric_plots(max_steps)
    
    # Keep the PyBullet window open until manually closed
    print("Press Enter to close the simulation...")
    input()
    
    env.close()
