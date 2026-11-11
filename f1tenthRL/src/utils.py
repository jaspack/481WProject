import os
import sys
import yaml
import argparse

# parse command-line arguments for map name
def parse_arguments():
        # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="F1TENTH Reinforcement Learning Framework")
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "evaluate"],
        help="Specify whether to 'train' the PPO model or 'evaluate' a saved policy."
    )
    
    return parser.parse_args()


# get configuration information for training
def load_training_configuration(configuration_path="config/training_configurations.yaml"):
    # check if the configuration file exists
    if not os.path.exists(configuration_path):
        # Check if the configuration file exists in the parent directory
        configuration_path = os.path.join("..", configuration_path)
        if not os.path.exists(configuration_path):
            print(f"Configuration file not found at {configuration_path}")
            sys.exit(1)
    with open(configuration_path, "r") as file:
        config = yaml.safe_load(file)
    check_training_configuration(config)
    return config

# check configuration for training
# TODO: Add configurations for model types, model name, number of episode, and action space
def check_training_configuration(config):
    # Define the required sections and keys for the configuration
    required_sections = {
            "environment": ["map_name", "max_episode_steps", "lidar_downsample_factor", "maps_base_directory"],
            "reward_weights": ["progress_weight", "velocity_weight", "smoothness_penalty", "collision_penalty"]
        }
    # Check for the presence of required sections and keys
    for section, keys in required_sections.items():
        if section not in config:
            # if the section is missing, print an error message and exit
            print(f"Missing required configuration section: {section}")
            sys.exit(1)
        for key in keys:
            if key not in config[section]:
                # if the key is missing, print an error message and exit
                print(f"Missing required configuration key '{key}' under section '{section}'")
                sys.exit(1)
    # Get the map name and base directory from the configuration
    map_name = config["environment"]["map_name"]
    maps_base_directory = config["environment"]["maps_base_directory"]

    # check if the map file specified in the configuration exists
    if not os.path.exists(f"{maps_base_directory}/{map_name}/{map_name}_map.png"):
        print(f"Map file not found for map '{map_name}' at path: {maps_base_directory}/{map_name}/{map_name}_map.png")
        sys.exit(1)
    # check if the map has a raceline csv file
    if not os.path.exists(f"{maps_base_directory}/{map_name}/{map_name}_raceline.csv"):
        print(f"Raceline file not found for map '{map_name}' at path: {maps_base_directory}/{map_name}/{map_name}_raceline.csv")
        sys.exit(1)