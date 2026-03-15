"""
Train.py - DroneAI Model Training

Trains a PPO (Proximal Policy Optimization) agent to fly the 2D drone.
Can train from scratch or continue training from an existing saved model.

Configuration:
    LOAD          - True to resume from an existing model, False for fresh training
    TIMESTEPS     - Number of environment steps to train before saving
    VISUALIZATION - True to render the simulation (slows training significantly)
    Model_Load_path - Path to load an existing model from (when LOAD=True)
    Model_path      - Path to save the trained model to

Usage:
    python3 Train.py
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Game import *


# ── Configuration ─────────────────────────────────────────────────────────────

Log_path = os.path.join('Training', 'Logs')              # TensorBoard log directory
Model_Load_path = os.path.join('Training', 'ModelEx')     # Path to load existing model from
Model_path = os.path.join('Training', 'ModelEx1')         # Path to save trained model to

LOAD = False            # Set True to continue training from Model_Load_path
TIMESTEPS = 3000000     # Total training timesteps before saving
VISUALIZATION = False   # Set True to render during training (not recommended)


# ── Environment setup ─────────────────────────────────────────────────────────

# Wrap the environment in a DummyVecEnv (required by Stable Baselines3)
env = DummyVecEnv([lambda: GameEnv(False)])

# Normalize observations and rewards for stable training.
# When loading, restore the saved normalization statistics.
if not LOAD: env = VecNormalize(env, clip_obs=math.inf, training=True)
else: env = VecNormalize.load(Model_Load_path+"env", env)


# ── Model setup ───────────────────────────────────────────────────────────────

# MlpPolicy: a multi-layer perceptron that maps observations → actions
if not LOAD: model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=Log_path)
else: model = PPO.load(Model_Load_path, env)


# ── Training ──────────────────────────────────────────────────────────────────

model.learn(total_timesteps=TIMESTEPS)
model.get_parameters()

# Save the trained model and the environment normalization statistics
env.save(Model_path+"env")
model.save(Model_path)
