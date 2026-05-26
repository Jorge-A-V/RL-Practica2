import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"
TABLA_CSV = "tabla_final.csv"
N_EPS_BASELINE = 20

if not os.path.exists(TABLA_CSV):
    raise FileNotFoundError(
        f"Falta {TABLA_CSV}"
    )

df_modelos = pd.read_csv(TABLA_CSV)
print("Resultados modelos desde csv")
print(df_modelos.to_string(index=False))

print(f"\nBaseline aleatorio ({N_EPS_BASELINE} episodios, mismos wrappers)")
rand_env = make_atari_env(ENV_ID, n_envs=1, seed=999)
rand_env = VecFrameStack(rand_env, n_stack=4)

rewards = []
for ep in range(N_EPS_BASELINE):
    obs = rand_env.reset()
    ep_r, done = 0.0, [False]
    while not done[0]:
        action = [rand_env.action_space.sample()]
        obs, r, done, _ = rand_env.step(action)
        ep_r += r[0]
    rewards.append(ep_r)
    print(f"  ep {ep:2d}: {ep_r:.1f}")
rand_env.close()

base_mean, base_std = float(np.mean(rewards)), float(np.std(rewards))
print(f"Aleatorio: {base_mean:.1f} +- {base_std:.1f}")

df_all = pd.concat([
    pd.DataFrame([{"algoritmo": "Aleatorio",
                   "reward": round(base_mean, 1),
                   "std": round(base_std, 1)}]),
    df_modelos,
], ignore_index=True)

print("\nTabla completa:")
print(df_all.to_string(index=False))
df_all.to_csv("tabla_final_con_baseline.csv", index=False)

colores = {"Aleatorio": "lightgray", "DQN": "steelblue", "PPO": "seagreen"}

fig, ax = plt.subplots(figsize=(7.5, 4.8))
xs = np.arange(len(df_all))
ax.bar(
    xs, df_all["reward"], yerr=df_all["std"],
    capsize=10,
    color=[colores.get(a, "gray") for a in df_all["algoritmo"]],
    edgecolor="black", linewidth=0.8,
    error_kw=dict(elinewidth=1.4, capthick=1.4),
)
ax.axhline(0, c="black", lw=0.6)

for x, m, s in zip(xs, df_all["reward"], df_all["std"]):
    ax.text(x, m + s + 5, f"{m:.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(xs)
ax.set_xticklabels(df_all["algoritmo"])
ax.set_ylabel(f"Recompensa media ({N_EPS_BASELINE} episodios)")
ax.set_title("SpaceInvaders-v5 - Evaluacion final")
ax.grid(axis="y", ls=":", alpha=0.5)

plt.tight_layout()
plt.savefig("eval_final.png", dpi=300)
plt.close(fig)
print("\nGuardado eval_final.png")
