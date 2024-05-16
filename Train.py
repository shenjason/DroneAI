import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Game import *


Log_path = os.path.join('Training', 'Logs')
Model_Load_path = os.path.join('Training', 'ModelEx')
Model_path = os.path.join('Training', 'ModelEx1')

LOAD = False
TIMESTEPS = 3000000
VISUALIZATION = False

env = DummyVecEnv([lambda: GameEnv(False)])

if not LOAD: env = VecNormalize(env, clip_obs=math.inf, training=True)
else: env = VecNormalize.load(Model_Load_path+"env", env)

if not LOAD: model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=Log_path)
else: model = PPO.load(Model_Load_path, env)

model.learn(total_timesteps=TIMESTEPS)
model.get_parameters()

env.save(Model_path+"env")
model.save(Model_path)
