"""
Load.py - DroneAI Model Viewer

Loads a trained PPO model and runs it in a visual pygame window so you can
watch the drone fly to targets in real time. The target follows the mouse
cursor (interactable mode).

Configuration:
    Model_path - Path to the saved model to load (without file extension)

Usage:
    python3 Load.py
"""

from Game import GameEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import pygame

# Path to the trained model (must match the path used during training)
Model_path = os.path.join('Training', 'ModelEx')

# Create the environment with visualization and mouse-interactive target
env = DummyVecEnv([lambda: GameEnv(visualization=True, interactable=True, cantimeout=False)])

# Load the saved observation/reward normalization statistics
env = VecNormalize.load(Model_path+"env", env)

# Load the trained PPO model
model = PPO.load(Model_path, env)

# ── Main loop ─────────────────────────────────────────────────────────────────

state = env.reset()
done = False

while not done:
    # Handle window close events
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            done = True

    # Let the trained model predict the next action
    action, thing = model.predict(state)

    # Step the environment (renders automatically since visualization=True)
    state, reward, done, info = env.step(action)

env.close()
