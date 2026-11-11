# code/evaluate.py
import os
import time
from stable_baselines3 import PPO
from .utils import load_training_configuration
from .f1tenth_environment import F1TenthEnvironment
# TODO: Add a way to specify the number of evaluation episodes in the configuration file
# Evaluates a trained model for the specified number of episodes and prints out the total reward for each episode.
def evaluate_agent(num_episodes=5):
    # Get the training configuration
    print("Loading training configuration...")
    config = load_training_configuration()

    env = F1TenthEnvironment(configurations=config)

    # Load the trained model
    # Currently the name of the model is hardcoded, need to add a way to specify the model name in the configuration file
    #TODO: Add a way to specify the model name in the configuration file
    model_path = os.path.join("models", "f1tenth_ppo_raceline.zip")
    # Check if the model file exists before trying to load it
    if not os.path.exists(model_path):
        # If the model file does not exist, print an error message and return
        print(f"Error: Trained model not found at {model_path}. Train the model first!")
        return
    # If the model file does exist, load the model
    print(f"Loading trained model from {model_path}...")
    # Currently, the model type is hardcoded to PPO
    #TODO: Add a way to specify the model type in the configuration file
    model = PPO.load(model_path)

    # Print out a message to indicate the start of the evaluation process
    print(f"\nStarting evaluation for {num_episodes} episodes...")

    # Run the evaluation for the specified number of episodes
    for episode in range(1, num_episodes + 1):
        # Reset the environment and get the initial observation
        obs, info = env.reset()
        # Set the initial values for terminated, truncated, episode_reward, and step_count
        terminated = False
        truncated = False
        episode_reward = 0.0
        step_count = 0

        # Run the episode until it is terminated or truncated
        while not (terminated or truncated):
            # Set deterministic=True to remove exploration noise and get the best action for evaluation
            action, _states = model.predict(obs, deterministic=True)

            # Take a step with the action and get the new observation
            obs, reward, terminated, truncated, info = env.step(action)
            # add the reward to the episode total and increment the step count
            episode_reward += reward
            step_count += 1
            # Render the environment to visualize the evaluation process
            env.render()

        # print out the total reward for the episode
        print(f"Episode {episode} finished | Steps: {step_count} | Total Reward: {episode_reward:.2f}")
    # Print a message to indicate the evaluation process is complete
    print("\nEvaluation complete.")