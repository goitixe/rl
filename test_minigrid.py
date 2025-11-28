import time
import gymnasium as gym
from pacman_env.pacman import PacmanEnv

if __name__ == "__main__":
    
    #tamaño 13 para coincidir con la logica del mapa
    env = PacmanEnv(render_mode="human", size=13, num_ghosts=3)
    
    obs, _ = env.reset()
    print("entorno pacman cargado")

    for i in range(1000):
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"step {i}: reward={reward}, over={terminated}")
        
        env.render()
        time.sleep(0.1)

        if terminated or truncated:
            print("reset")
            env.reset()

    env.close()