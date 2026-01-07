import gymnasium as gym
import torch as th
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from gymnasium.wrappers import FlattenObservation
from pacman_env.custom_env import ProPacmanEnv

#configuracion intento rapido
VISION_MODE = "global" 
RUN_NAME = f"Pacman_Curriculum_Fast"
LOG_DIR = "./logs/"
MODELS_DIR = f"./models/{RUN_NAME}"
SEED = 42

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

POLICY_KWARGS = dict(
    activation_fn=th.nn.Tanh,
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

def make_env_stage(n_ghosts, mode):
    def _init():
        env = ProPacmanEnv(size=19, num_ghosts=n_ghosts, render_mode="rgb_array")
        if mode == "global":
            env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)
        return env
    return _init

def train_stage(stage_name, n_ghosts, timesteps, base_model_path=None):
    print(f"entrenando {stage_name} con {n_ghosts} fantasmas")
    vec_env = make_vec_env(make_env_stage(n_ghosts, VISION_MODE), n_envs=8)

    if base_model_path is None:
        print("modelo nuevo")
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.01,
            seed=SEED
        )
    else:
        print(f"cargando {base_model_path}")
        model = PPO.load(base_model_path, env=vec_env)
    
    full_log_name = f"{RUN_NAME}_{stage_name}"
    model.learn(total_timesteps=timesteps, tb_log_name=full_log_name, reset_num_timesteps=False)
    
    save_path = f"{MODELS_DIR}/{stage_name}"
    model.save(save_path)
    print(f"guardado en {save_path}")
    vec_env.close()
    return f"{save_path}.zip"

if __name__ == "__main__":
    #fase 1: muy rapida para no viciar al agente
    last = train_stage("Stage1_Easy", 0, 50000, None)

    #fase 2: introduccion al peligro
    last = train_stage("Stage2_Med", 2, 150000, last)

    #fase 3: entrenamiento fuerte
    last = train_stage("Stage3_Hard", 4, 300000, last)
    
    final_path = f"{MODELS_DIR}/final_model"
    if os.path.exists(f"{final_path}.zip"):
        os.remove(f"{final_path}.zip")
    os.rename(last, f"{final_path}.zip")
    print("curriculum rapido terminado")