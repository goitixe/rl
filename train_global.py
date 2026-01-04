import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from gymnasium.wrappers import FlattenObservation
from pacman_env.pacman import PacmanEnv

#configuracion
TIMESTEPS = 300000
RUN_NAME = "Pacman_Global_v2"

#funcion para crear el entorno global
def make_env():
    #creamos el entorno basico
    env = PacmanEnv(size=13, num_ghosts=2)
    #wrapper para ver todo el mapa (vision global)
    env = FullyObsWrapper(env)
    #convertir a numeros
    env = ImgObsWrapper(env)
    #aplanar para usar mlp (igual que en el local)
    env = FlattenObservation(env)
    return env

if __name__ == "__main__":
    #4 entornos paralelos para que aprenda mas rapido
    vec_env = make_vec_env(make_env, n_envs=4)

    #configuracion de red neuronal optimizada (256 neuronas)
    #usamos la misma que en local para que la comparacion sea justa
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    #usamos mlppolicy (red densa vista en clase)
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01
    )

    print(f"empezando a entrenar {RUN_NAME}...")
    
    #guardado de seguridad en carpeta models
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=f"./models/{RUN_NAME}")

    model.learn(
        total_timesteps=TIMESTEPS, 
        tb_log_name=RUN_NAME, 
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save(f"models/{RUN_NAME}/final_model")
    print("entrenamiento global acabado.")


#---------------------------------------------------------
#CODIGO VIEJO
#guardado para la memoria
#---------------------------------------------------------
#import gymnasium as gym
#from stable_baselines3 import PPO
#import os
#from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
#from pacman_env.pacman import PacmanEnv

##configuracion
#ALGORITHM = "Pacman_Global"
#models_dir = "models/" + ALGORITHM
#log_dir = "logs"

#if not os.path.exists(models_dir):
#    os.makedirs(models_dir)

#if not os.path.exists(log_dir):
#    os.makedirs(log_dir)

##crear el entorno directamente
#env = PacmanEnv(size=13, num_ghosts=2)

##wrappers para vision global
#env = FullyObsWrapper(env)
#env = ImgObsWrapper(env)

##instanciar el agente
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0003)

#TIMESTEPS = 100000

#print("iniciando entrenamiento: " + ALGORITHM)

#model.learn(total_timesteps=TIMESTEPS, tb_log_name=ALGORITHM)

##guardado
#model.save(models_dir + "/" + str(TIMESTEPS))
#print("modelo guardado")

#env.close()