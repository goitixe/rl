import time
import os
import gymnasium as gym
from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from gymnasium.wrappers import FlattenObservation
from pacman_env.pacman import PacmanEnv

#configuracion: cambia a "LOCAL" o "GLOBAL"
MODO = "LOCAL"

#seleccionamos el modelo correcto segun tus carpetas reales
if MODO == "LOCAL":
    RUN_NAME = "Pacman_Local_MLP_Big" # <--- CORREGIDO: Este es el nombre real de tu carpeta local
else:
    RUN_NAME = "Pacman_Global_v2"     # <--- CORREGIDO: Este es el nombre real de tu carpeta global

MODEL_PATH = f"models/{RUN_NAME}/final_model"

if __name__ == "__main__":
    
    #configuramos el entorno segun el modo
    if MODO == "LOCAL":
        #vision recortada 7x7
        env = PacmanEnv(render_mode="human", size=13, num_ghosts=2, agent_view_size=7)
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)
    else:
        #vision completa
        env = PacmanEnv(render_mode="human", size=13, num_ghosts=2)
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)

    print(f"cargando entorno {MODO} y modelo {RUN_NAME}...")

    #chequeo por si acaso
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"error: no encuentro el archivo {MODEL_PATH}.zip")
        exit()

    model = PPO.load(MODEL_PATH, env=env)
    
    #probamos 5 episodios
    for i in range(5):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        print(f"--- episodio {i+1} ---")

        while not (terminated or truncated):
            #usamos el modelo entrenado
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            #visualizacion
            env.render()
            time.sleep(0.05)

        print(f"fin episodio: reward={total_reward}")
        time.sleep(0.5)

    env.close()