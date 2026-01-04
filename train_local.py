import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers import FlattenObservation
from pacman_env.pacman import PacmanEnv
import os
import torch as th

# ==============================================================================
#  CONFIGURACIÓN DE LA VERSIÓN OPTIMIZADA (PASO 2)
# ==============================================================================
TIMESTEPS = 300000 
RUN_NAME = "Pacman_Local_MLP_Big" # Nombre clave para diferenciarlo en TensorBoard
LOG_DIR = "./logs/"
MODELS_DIR = f"./models/{RUN_NAME}"

# Crear directorios si no existen
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def make_env():
    # 1. Entorno Base (Visión Local 7x7)
    # Reducimos la visión a 7x7 para que sea más fácil de procesar localmente
    env = PacmanEnv(size=13, num_ghosts=2, agent_view_size=7)
    
    # 2. Wrapper de Imagen (Convierte a matriz numérica)
    env = ImgObsWrapper(env)
    
    # 3. Aplanar (Flatten) - MEJORA CRÍTICA
    # Transforma la matriz 7x7x3 en un vector de 147 números.
    # Necesario para usar MlpPolicy de forma eficiente.
    env = FlattenObservation(env)
    return env

if __name__ == "__main__":
    # Usamos 4 entornos en paralelo para acelerar la recolección de datos
    vec_env = make_vec_env(make_env, n_envs=4)

    # --- OPTIMIZACIÓN "BIG BRAIN" (JUSTIFICACIÓN TÉCNICA) ---
    # Como usamos MlpPolicy (según teoría) pero el problema es complejo,
    # aumentamos la capacidad de la red neuronal.
    # Pasamos de [64, 64] (default) a [256, 256].
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        "MlpPolicy",       # Política estándar (Red Neuronal Artificial)
        vec_env,
        policy_kwargs=policy_kwargs, # Inyectamos la arquitectura mejorada
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01      # Coeficiente de entropía para fomentar exploración
    )

    print(f"--- INICIANDO ENTRENAMIENTO (OPTIMIZADO): {RUN_NAME} ---")
    
    # Guardado de seguridad cada 50k pasos
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=MODELS_DIR)

    model.learn(
        total_timesteps=TIMESTEPS, 
        tb_log_name=RUN_NAME, 
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save(f"{MODELS_DIR}/final_model")
    print("--- ENTRENAMIENTO FINALIZADO Y MODELO GUARDADO ---")


"""
# ==============================================================================
#  ARCHIVO HISTÓRICO: VERSIÓN LEGACY (INTENTO ORIGINAL)
#  Código original utilizado en las primeras pruebas. 
#  Se conserva para análisis comparativo de rendimiento.
# ==============================================================================

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from minigrid.wrappers import ImgObsWrapper
from pacman_env.pacman import PacmanEnv

#configuracion
TIMESTEPS = 100000
RUN_NAME = "Pacman_Local_13x13"

def make_env():
    #entorno con tamaño 13
    env = PacmanEnv(size=13, num_ghosts=2)
    #wrapper necesario para ppo
    env = ImgObsWrapper(env)
    return env

if __name__ == "__main__":
    #entorno vectorizado
    vec_env = make_vec_env(make_env, n_envs=1)

    #usamos mlppolicy para evitar el error de kernel size
    #la imagen local es muy pequeña para cnn estandar
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        tensorboard_log="./logs/",
        learning_rate=0.0003
    )

    print("iniciando entrenamiento local...")
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=RUN_NAME)
    
    model.save(f"models/{RUN_NAME}")
    print("modelo guardado")
"""