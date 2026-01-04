import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from minigrid.wrappers import ImgObsWrapper
#sin fullyobs para vision local
from gymnasium.wrappers import FlattenObservation

#importacion adaptada a la nueva ubicacion
from pacman_env.custom_env import ProPacmanEnv

#configuracion v3 local
TIMESTEPS = 600000
RUN_NAME = "Pacman_Local_v3"

def make_env():
    env = ProPacmanEnv(size=19, num_ghosts=2, render_mode="rgb_array")
    #vision local por defecto (7x7)
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)
    return env

if __name__ == "__main__":
    vec_env = make_vec_env(make_env, n_envs=8)

    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

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
    
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=f"./models/{RUN_NAME}")

    print(f"iniciando entrenamiento local: {RUN_NAME}")
    
    model.learn(
        total_timesteps=TIMESTEPS, 
        tb_log_name=RUN_NAME, 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    model.save(f"models/{RUN_NAME}/final_model")
    print("local v3 finalizado")