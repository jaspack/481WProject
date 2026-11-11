



import os
import numpy as np

# Calculates the reward for the F1Tenth simulator based on the current state of the car and its actions
class F1TenthRewardCalculator:
    def __init__(self, configurations):
        # Get the configuration information
        self.configurations = configurations
        # Get the reward weights from the configuration dictionary
        reward_configurations = configurations.get("reward_weights")
        # Set the initial values to 0, these are used to track waypoint progress and the distance traveled
        self.previous_index = 0
        self.nearest_index = 0
        
        # Get each weight
        self.progress_weight = reward_configurations.get("progress_weight", 1.0)
        self.velocity_weight = reward_configurations.get("velocity_weight", 0.5)
        self.collision_penalty = reward_configurations.get("collision_penalty", -100.0)
        # This one is not currently used, could be used to penalize jerky driving
        self.smoothness_penalty = reward_configurations.get("smoothness_penalty", 0.1)
        
        # Get the base directory for the maps from the configuration dictionary
        self.maps_base_directory = configurations.get("environment").get("maps_base_directory")
        # get the map name from the configuration dictionary
        self.map_name = configurations.get("environment").get("map_name")
        # Get the raceline csv file for the specified map
        self.raceline = self._get_raceline(self.map_name)

        # Create an array to hold the distances from the current position to each waypoint on the raceline
        self.distances = np.zeros(self.raceline.shape[0])
        # Initialize the previous s value to zero for the first step
        self.previous_s = 0.0

        # Initialize the previous action to None for the first step
        self.previous_action = [0.0, 0.0]

    # Functions
    # Compute the reward based on the raw observation and the action taken
    def compute_reward(self, raw_obs, action):
        # first, initialize the reward to zero
        reward = 0.0

        # Next, check how fast the car is moving and apply the velocity weight to the reward
        velocity = action[1]
        reward += (velocity - 0.1) * self.velocity_weight
    
        # then, calculate the progress along the raceline
        progress_delta = self._calculate_raceline_progress(raw_obs)
        # multiply the change in progress by the progress weight and add to the reward
        reward += progress_delta * self.progress_weight
    
        # keep track of the current action for the next step's smoothness calculation
        self.previous_action = action
    
        # then, check if the car has collided with any obstacles
        if raw_obs['collisions'][0] > 0:
            # if there is a collision, apply the collision penalty to the reward
            reward += self.collision_penalty  # the collision penalty is negative, so we add it to the reward to decrease it
        # finally, return the computed reward as a float
        return float(reward)

    # calculate the change in progress along the raceline based on the raw observation
    def _calculate_raceline_progress(self, raw_obs):
        # Get the current position of the car from the raw observation
        current_position = (raw_obs['poses_x'][0], raw_obs['poses_y'][0])
        # Find the nearest waypoint index on the raceline
        self.nearest_index = self._find_nearest_waypoint_index(current_position)
        # Get the distance along the raceline at that index
        current_s = self.raceline[self.nearest_index, 0]
        # calculate the 
        delta_s = current_s - self.previous_s
        if delta_s < 0:
            debug = True
        # Store the current distance along the raceline for the next step's calculation
        self.previous_s = current_s
        # Return the change in distance along the raceline as a float
        return float(delta_s)

    # find the nearest waypoint index on the raceline to the current position of the car
    def _find_nearest_waypoint_index(self, current_position):
        # Calculate Euclidean distance from car to every point on the raceline and store it in the distances array
        for i in range(self.raceline.shape[0]):
            # get the x coordinate of the raceline point
            x_position = self.raceline[i,1]
            # get the y coordinate of the raceline point
            y_position = self.raceline[i,2]
            # calculate the distance from the current position to the raceline point
            self.distances[i] = np.sqrt(((current_position[0] - x_position) ** 2) + ((current_position[1] - y_position) ** 2))
        # find the index of the minimum distance
        return np.argmin(self.distances)

    # get the raceline data from the CSV file for the specified map
    def _get_raceline(self, map_name):
        # Construct the path to the raceline CSV file based on the map name
        raceline_path = f"{self.maps_base_directory}/{map_name}/{map_name}_raceline.csv"

        # Load the raceline data from the CSV file
        raceline_data = np.loadtxt(raceline_path, delimiter=',', skiprows=1)

        # Return the loaded raceline data
        return raceline_data

    def _get_initial_poses(self):
        # Get the starting position from the first waypoint of the raceline
        x0 = self.raceline[0, 1]
        y0 = self.raceline[0, 2]
        
        # Get the next waypoint
        x1 = self.raceline[1, 1]
        y1 = self.raceline[1, 2]
        # Calculate the heading angle based on the two waypoints
        theta = np.arctan2(y1 - y0, x1 - x0)
        
        # Return the initial pose
        return np.array([[x0, y0, theta]], dtype=np.float32)