import os
import sys
import numpy as np
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageSequenceClip

from stable_baselines3 import DQN, PPO, A2C

# seed randm
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

max_steps = 1000
frames_final = 30  # 30 frmas 1 seg final

ENV_ID = "LunarLander-v3"
OUT_DIR = "videos_lunar"
os.makedirs(OUT_DIR, exist_ok=True)

algorimtos = [
    ("DQN", DQN, "logs_dqn/dqn_lunarlander"),
    ("PPO", PPO, "logs_ppo/ppo_lunarlander"),
    ("A2C", A2C, "logs_a2c/a2c_lunarlander"),
]

FONT = ImageFont.load_default()

def anotar(rgb, nombre, reward, done):
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    bar_h = 38
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=(0, 0, 0))
    color = (255, 90, 90) if done else (255, 255, 255)
    tag = f"{nombre}: {reward:7.1f}" + ("  [FIN]" if done else "")
    draw.text((10, 8), tag, fill=color, font=FONT)
    return np.array(img)


def videos():
    estados = []
    for nombre, clase, ruta in algorimtos:
        if not os.path.exists(ruta + ".zip"):
            raise FileNotFoundError(f"no existe {ruta}.zip")
        env = gym.make(ENV_ID, render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)
        estados.append({
            "name": nombre,
            "env": env,
            "obs": obs,
            "model": clase.load(ruta),
            "done": False,
            "total_r": 0.0,
            "last_frame": None,
        })

    frames = []
    for step in range(max_steps):
        paneles = []
        for s in estados:
            if not s["done"]:
                action, _ = s["model"].predict(s["obs"], deterministic=True)
                s["obs"], r, term, trunc, _ = s["env"].step(action)
                s["total_r"] += r
                if term or trunc:
                    s["done"] = True
            s["last_frame"] = s["env"].render()
            paneles.append(anotar(s["last_frame"], s["name"], s["total_r"], s["done"]))

        frames.append(np.hstack(paneles))

        if all(s["done"] for s in estados):
            # frme extra final
            ultimo = frames[-1]
            for _ in range(frames_final):
                frames.append(ultimo)
            break

    for s in estados:
        s["env"].close()
        print(f"  {s['name']}: reward final = {s['total_r']:.1f}")

    out = os.path.join(OUT_DIR, "comparativa_3algos.mp4")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out, codec="libx264", audio=False, logger=None)
    print(f"\nGuardado {out} ({len(frames)} frames, {len(frames)/fps:.1f}s)")


videos()