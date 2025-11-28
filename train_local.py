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