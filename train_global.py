import gymnasium as gym
from stable_baselines3 import PPO
import os
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from pacman_env.pacman import PacmanEnv

#configuracion
ALGORITHM = "Pacman_Global"
models_dir = "models/" + ALGORITHM
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#crear el entorno directamente
env = PacmanEnv(size=13, num_ghosts=2)

#wrappers para vision global
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

#instanciar el agente
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0003)

TIMESTEPS = 100000

print("iniciando entrenamiento: " + ALGORITHM)

model.learn(total_timesteps=TIMESTEPS, tb_log_name=ALGORITHM)

#guardado
model.save(models_dir + "/" + str(TIMESTEPS))
print("modelo guardado")

env.close()