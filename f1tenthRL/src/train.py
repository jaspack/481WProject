import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from .utils import load_training_configuration
from .f1tenth_environment import F1TenthEnvironment

# Trains a PPO agent and saves the trained model to a file. If a previously trained model exists, 
# it will continue training from that model.
# TODO: Add a way to specify the model type and name in the configuration file
def train_agent():
    # Print a message to indicate the start of the training process
    print("Loading training configuration...")
    # First, load the training configuration
    config = load_training_configuration()

    # Creates and returns a monitored F1TenthEnvironment instance for training or evaluation.
    #def make_env():
    #    env = F1TenthEnvironment(config)
    #    return Monitor(env)

    # Wrap the environment in a DummyVecEnv
    #vec_env = DummyVecEnv([make_env])
    vec_env = make_vec_env(
        lambda: F1TenthEnvironment(config),
        n_envs=4,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )


    # Extract total timesteps from config if available, default to 100k
    total_timesteps = config.get("environment", {}).get("total_timesteps", 100_000)

    
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "f1tenth_ppo_raceline.zip")

    # Check if a previously trained model exists to resume training
    if os.path.exists(model_path):
        # If the model exists, load it
        print(f"Loading existing model from {model_path} for continued training...")
        model = PPO.load(model_path, env=vec_env, verbose=1)
    else:
        # Otherwise, create a new PPO model
        print("No existing model found. Initializing a new PPO model...")
        model = PPO(
            device="cpu",
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./f1tenth_tensorboard/"
        )

    # Print a message to indicate the start of the training loop
    print(f"Starting training for {total_timesteps} timesteps...")
    # Train the model for the specified number of timesteps
    model.learn(total_timesteps=total_timesteps)

    # Save the trained policy weights
    model_path = os.path.join("models", "f1tenth_ppo_raceline")
    model.save(model_path)
    # Print a message to indicate the training is complete and the model has been saved
    print(f"Training complete. Model successfully saved to {model_path}.zip")
