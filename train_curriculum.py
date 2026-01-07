import gymnasium as gym
import torch as th
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from gymnasium.wrappers import FlattenObservation

#importamos tu entorno
from pacman_env.custom_env import ProPacmanEnv

#configuracion para intentar arreglar el global
VISION_MODE = "global" 
RUN_NAME = f"Pacman_Curriculum_{VISION_MODE}"
LOG_DIR = "./logs/"
MODELS_DIR = f"./models/{RUN_NAME}"
SEED = 42

#crear carpetas
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

#misma red que en los experimentos v3
POLICY_KWARGS = dict(
    activation_fn=th.nn.Tanh,
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

def make_env_stage(n_ghosts, mode):
    #funcion para crear el entorno segun dificultad
    def _init():
        #mapa 19x19 siempre
        env = ProPacmanEnv(size=19, num_ghosts=n_ghosts, render_mode="rgb_array")
        
        if mode == "global":
            env = FullyObsWrapper(env)
        
        #wrappers necesarios
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)
        return env
    return _init

def train_stage(stage_name, n_ghosts, timesteps, base_model_path=None):
    print(f"entrenando {stage_name} con {n_ghosts} fantasmas")
    
    #8 entornos para ir rapido
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
    
    #entrenamos sin resetear pasos para ver bien la grafica
    full_log_name = f"{RUN_NAME}_{stage_name}"
    model.learn(
        total_timesteps=timesteps, 
        tb_log_name=full_log_name, 
        reset_num_timesteps=False, 
        progress_bar=True
    )
    
    save_path = f"{MODELS_DIR}/{stage_name}"
    model.save(save_path)
    print(f"guardado en {save_path}")
    
    vec_env.close()
    return f"{save_path}.zip"

if __name__ == "__main__":
    #fase 1 sin fantasmas
    last_model = train_stage(
        stage_name="Stage1_NoGhosts", 
        n_ghosts=0, 
        timesteps=150000, 
        base_model_path=None
    )

    #fase 2 con 2 fantasmas
    last_model = train_stage(
        stage_name="Stage2_Medium", 
        n_ghosts=2, 
        timesteps=300000, 
        base_model_path=last_model
    )

    #fase 3 con 4 fantasmas
    last_model = train_stage(
        stage_name="Stage3_Hard", 
        n_ghosts=4, 
        timesteps=400000, 
        base_model_path=last_model
    )
    
    #guardado final
    final_path = f"{MODELS_DIR}/final_model"
    if os.path.exists(f"{final_path}.zip"):
        os.remove(f"{final_path}.zip")
    os.rename(last_model, f"{final_path}.zip")
    
    print("curriculum terminado")