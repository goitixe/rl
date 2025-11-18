import gymnasium as gym
import minigrid

env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")

obs, info = env.reset()
print("Obs type:", type(obs))
print("Obs content:", obs)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("Step -> reward:", reward, "terminated:", terminated, "truncated:", truncated)

env.close()
print("MiniGrid funciona correctamente ✅")
