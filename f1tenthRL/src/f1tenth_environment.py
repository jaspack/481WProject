import gym as legacy_gym
import gymnasium as new_gym
import numpy as np
from .f1tenth_reward_calculator import F1TenthRewardCalculator

# Creates a custom Gymnasium environment for the F1Tenth simulator
class F1TenthEnvironment(new_gym.Env):
    def __init__(self, configurations, lidar_downsample_factor=8):
        super().__init__()
        self.configurations = configurations
        # First, initialize the legacy gym environment with the specified map
        map_name = configurations.get("environment").get("map_name")
        map_path = f"maps/{map_name}/{map_name}_map"
        self.f110simulation = legacy_gym.make('f110_gym:f110-v0', map=map_path, num_agents=1)
        
        # Then, instantiate a reward calculator
        self.reward_calculator = F1TenthRewardCalculator(configurations)
        self.initial_poses = self.reward_calculator._get_initial_poses()

        # set the current step counter to 0 and maximum steps to the value in the configuration file
        self.current_step = 0
        self.max_steps = configurations.get("environment").get("max_episode_steps")

        # set the lidar downsample factor
        # Basically, the downsample factor determines how many beams are skipped
        self.lidar_downsample_factor = lidar_downsample_factor

        # Calculate the total number of features in the observation space
        raw_beam_count = 1080
        self.total_features = raw_beam_count // self.lidar_downsample_factor

        # Define a maximum lidar range for clipping purposes
        self.max_lidar_range = 30.0  # meters
        
        # Create an observation space using the max range and the total number of beams 
        self.observation_space = new_gym.spaces.Box(
            low=0.0,
            high=self.max_lidar_range,
            shape=(self.total_features,),
            dtype=np.float32
        )

        # create an action space for the car, contains steering and velocity
        # currently these are hardcoded, they should be specified in the configuration file
        #TODO: Add a way to specify the action space in the configuration file
        self.action_space = new_gym.spaces.Box(
            low=np.array([-0.4189, 0.1], dtype=np.float32),  # Min steering = -24 deg, Min speed 0.1 m/s
            high=np.array([0.4189, 7.0], dtype=np.float32), # Max steering = 24 deg, Max speed = 7 m/s
            shape=(2,),
            dtype=np.float32
        )
    # Resets the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Call the f1tenth simulator reset function with the initial poses to reset the environment
        raw_obs, empty_reward, done, info = self.f110simulation.reset(self.initial_poses)
        # process the raw observation from the f1tenth simulator
        obs = self._process_obs(raw_obs)
        # reset the current step counter
        self.current_step = 0
        # return the processed observation and an empty info dictionary
        # Re-instantiate the reward calculator to reset its internal state for the new episode
        self.reward_calculator = F1TenthRewardCalculator(self.configurations)
        return obs, {}
    # Steps the environment given an action, returns the new observation
    def step(self, action):
        # Extract the steering and velocity from the action array
        steering = action[0]
        velocity = action[1]
        # create an action array with the specified steering and velocity for the f1tenth simulator
        sim_action = np.array([[steering, velocity]], dtype=np.float32)
        # increment the step count
        self.current_step += 1
        # Call the f1tenth simulator step function with the action and get the raw observation
        # Reward is empty because f1tenth doesn't provide one
        raw_obs, empty_reward, done, info = self.f110simulation.step(sim_action)
        # Process the raw observation from the f1tenth simulator
        obs = self._process_obs(raw_obs)
        
        # Compute the reward based on the raw observation and the action taken
        reward = self.reward_calculator.compute_reward(raw_obs, action)
        # Check if there was a collision, if there was, terminated is true, otherwise false.
        terminated = bool(raw_obs['collisions'][0] > 0)
        # Check if the maximum number of steps has been reached, if it has, truncated is true, otherwise false.
        truncated = self.current_step >= self.max_steps
        # Return the processed observation, computed reward, terminated flag, truncated flag, and an empty info dictionary
        if terminated:
            debug = True
        return obs, reward, terminated, truncated, info

    # Renders the f1tenth simulator environment
    def render(self):
        return self.f110simulation.render(mode='human')
    # Closes the f1tenth simulator environment
    def close(self):
        self.f110simulation.close()
    # Processes the raw observation from the f1tenth simulator and returns a downsampled and clipped lidar scan
    def _process_obs(self, raw_obs):

        # get the lidar scan from the raw observation
        lidar_scan = raw_obs['scans'][0]
        
        # Downsample the lidar scan by taking every nth value, where n is the downsample factor
        lidar_downsampled = lidar_scan[::self.lidar_downsample_factor]
        
        # Clip any out-of-range or infinite values to a clean maximum limit
        lidar_clipped = np.clip(lidar_downsampled, 0.0, self.max_lidar_range)
        
        # Return as a flat 1D float32 array for Stable Baselines3 compatibility
        return lidar_clipped.astype(np.float32)