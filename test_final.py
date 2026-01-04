import gymnasium as gym
import time
from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from gymnasium.wrappers import FlattenObservation
from pacman_env.custom_env import ProPacmanEnv

#configuracion, se cambia el model path pa probar local y yavale 
MODEL_PATH = "models/Pacman_Global_v3/final_model" #asegurate que coincide con el train

def make_test_env():
    #importante: render_mode human para ver la ventana grafica
    env = ProPacmanEnv(size=19, num_ghosts=2, render_mode="human")
    
    #wrappers identicos al entrenamiento global
    #si pruebas el modelo local, comenta la linea de FullyObsWrapper
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)
    return env

if __name__ == "__main__":
    try:
        print(f"cargando modelo desde: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print("error: no se encuentra el archivo del modelo. espera a que termine el entrenamiento.")
        exit()

    #creamos entorno de visualizacion
    env = make_test_env()

    #jugamos 5 partidas de demostracion
    for i in range(5):
        print(f"--- iniciando partida {i+1} ---")
        
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (terminated or truncated):
            #deterministic=True hace que la ia use su mejor jugada siempre
            action, _state = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            #actualizar ventana
            env.render()
            
            #pausa para visualizacion fluida (ajustar si va muy rapido)
            time.sleep(0.05)

        print(f"fin partida {i+1}: recompensa={total_reward:.2f}, pasos={steps}")
        time.sleep(1.0) #pausa entre partidas

    env.close()