import os
import sys
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

ENV_ID = "ALE/SpaceInvaders-v5"
VIDEO_LENGTH = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

MODELOS = [
    ("dqn", DQN, "logs_dqn/dqn_spaceinvaders", "videos_dqn"),
    ("ppo", PPO, "logs_ppo/ppo_spaceinvaders", "videos_ppo"),
]

for nombre, clase, ruta_modelo, video_dir in MODELOS:
    os.makedirs(video_dir, exist_ok=True)
    print(f"\n{nombre.upper()} Modelo {ruta_modelo}")

    env = make_atari_env(
        ENV_ID, n_envs=1,
        env_kwargs={"render_mode": "rgb_array"},
        wrapper_kwargs={"terminal_on_life_loss": False, "clip_reward": False},
    )
    env = VecFrameStack(env, n_stack=4)

    env = VecVideoRecorder(
        env, video_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=VIDEO_LENGTH,
        name_prefix=f"{nombre}_spaceinvaders",
    )

    model = clase.load(ruta_modelo)
    obs = env.reset()
    total_r = 0.0

    for step in range(VIDEO_LENGTH):
        action, _ = model.predict(obs, deterministic=False)
        obs, r, done, _ = env.step(action)
        total_r += r[0]

        if done[0]:
            print(f"Episodio terminado stp {step}, score real {total_r:.0f}")
            break
    else:
        print(f"{VIDEO_LENGTH} frames, score real acmlado {total_r:.0f}")

    env.close()
    
    del env, model, obs
    
    print(f"Video {video_dir}/")