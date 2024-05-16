from Game import GameEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import pygame

Model_path = os.path.join('Training', 'ModelEx')

env = DummyVecEnv([lambda: GameEnv(visualization=True, interactable=True, cantimeout=False)])

env = VecNormalize.load(Model_path+"env", env)

model = PPO.load(Model_path, env)

state = env.reset()
done = False

while not done:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            done = True
    action, thing = model.predict(state)

    state, reward, done, info = env.step(action)
env.close()
