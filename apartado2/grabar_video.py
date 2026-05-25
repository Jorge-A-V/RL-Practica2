import os
import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import torch

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_EPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MODEL_PATH = "logs_ppo/best/best_model.zip"
VIDEO_DIR = "videos_lunar"

os.makedirs(VIDEO_DIR, exist_ok=True)

env = gym.make("LunarLander-v3", render_mode="rgb_array")

env = RecordVideo(
    env, 
    video_folder=VIDEO_DIR,
    episode_trigger=lambda i: True,
    name_prefix="ppo_lunarlander",
    disable_logger=True,
)

print(f"Modelo: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

rewards = []
for ep in range(N_EPS):
    obs, _ = env.reset(seed=SEED + ep)
    done = False
    total_r = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        total_r += r
        steps += 1
        done = term or trunc

    rewards.append(total_r)
    print(f"Episodio {ep}: rec {total_r:.1f} (Steps: {steps})")

env.close()
media = sum(rewards) / len(rewards)
print(f"\nMedia {N_EPS} episodios: {media:.1f}")
print(f"Videos {VIDEO_DIR}/")