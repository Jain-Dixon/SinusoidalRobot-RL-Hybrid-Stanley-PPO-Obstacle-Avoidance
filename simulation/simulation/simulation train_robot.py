"""
train_robot.py
==============
Curriculum training for the sinusoidal robot environment.

Usage
-----
  python train_robot.py --mode stage1                 # ~1.5M steps
  python train_robot.py --mode stage2                 # ~2.0M steps
  python train_robot.py --mode test     --stage 2     # visual test
  python train_robot.py --mode showcase --stage 2     # perfect eps only
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback)
from stable_baselines3.common.monitor import Monitor
from robot_env import SinusoidalRobotEnv   # noqa: F401 — triggers gym.register


# ════════════════════════════════════════════════════════════════
# TRAINING MONITOR
# Prints one summary line every 20 completed episodes.
# Tracks: goals, collisions, boundary exits separately.
# ════════════════════════════════════════════════════════════════
class TrainingMonitor(BaseCallback):
    def __init__(self, stage):
        super().__init__()
        self.stage      = stage
        self.rewards    = []
        self.goals      = 0
        self.collisions = 0
        self.boundaries = 0   # |y| > 2.5 exits

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            ep_r = info["episode"]["r"]
            self.rewards.append(ep_r)
            if info.get("goal_reached"):  self.goals      += 1
            if info.get("collision"):     self.collisions += 1
            if info.get("boundary_exit"): self.boundaries += 1

            n = len(self.rewards)
            if n % 20 != 0:
                continue

            avg_r    = np.mean(self.rewards[-20:])
            best_r   = np.max(self.rewards)
            goal_pct = self.goals      / n * 100
            coll_pct = self.collisions / n * 100
            bnd_pct  = self.boundaries / n * 100
            tag = ("🟢 EXCELLENT" if avg_r > 2000 else
                   "🟡 LEARNING"  if avg_r > 500  else
                   "🔵 IMPROVING" if avg_r > 0    else
                   "⚪ EXPLORING")
            print(
                f"[S{self.stage}] Ep:{n:5d} | "
                f"Steps:{self.num_timesteps:9,} | "
                f"AvgR:{avg_r:8.1f} | Best:{best_r:8.1f} | "
                f"Goals:{goal_pct:5.1f}% | "
                f"Coll:{coll_pct:4.1f}% | "
                f"OOB:{bnd_pct:4.1f}% | {tag}")
        return True


# ════════════════════════════════════════════════════════════════
# STAGE 1 — SINE PATH FOLLOWING  (no obstacles)
# ════════════════════════════════════════════════════════════════
def train_stage1(timesteps=1_500_000):
    """
    Robot learns to follow the curvy sine path.
      amplitude = 0.50 m,  frequency = 0.15 cycles/m
    Target: Goal rate > 80%,  avg CTE < 0.10 m
    Expected reward range: good episode ~1500-3000, bad ~-2000
    """
    print("=" * 72)
    print("  STAGE 1 — SINE PATH FOLLOWING  (no obstacles)")
    print("  Sine: amplitude=0.50 m,  frequency=0.15")
    print("  Target: Goals > 80%,  avg CTE < 0.10 m")
    print("  Expected AvgR when learning well: > 500")
    print("=" * 72)

    env      = Monitor(gym.make("SinusoidalRobot-v1-stage1", render_mode=None))
    eval_env = Monitor(gym.make("SinusoidalRobot-v1-stage1", render_mode=None))
    check_env(env)
    print("✓ Env OK\n")

    os.makedirs("models/stage1", exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        tensorboard_log="./logs/",
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[
            CheckpointCallback(
                save_freq=25_000, save_path="models/stage1/",
                name_prefix="robot_s1", verbose=0),
            EvalCallback(
                eval_env,
                best_model_save_path="models/stage1/",
                eval_freq=10_000, n_eval_episodes=20,
                deterministic=True, render=False, verbose=1),
            TrainingMonitor(stage=1),
        ],
        progress_bar=True)

    model.save("models/stage1/final")
    print("\n✅ STAGE 1 DONE")
    print("   models/stage1/best_model.zip")
    print("   Next: python train_robot.py --mode stage2\n")
    env.close()
    eval_env.close()


# ════════════════════════════════════════════════════════════════
# STAGE 2 — OBSTACLE AVOIDANCE
# ════════════════════════════════════════════════════════════════
def train_stage2(timesteps=2_000_000):
    """
    Fine-tunes Stage 1 model with 2 obstacles on path at x=5, x=10.
    Robot must: detect → steer LEFT → clear → snap back RIGHT → continue.
    Target: Goal rate > 60%,  Coll = 0%
    """
    print("=" * 72)
    print("  STAGE 2 — OBSTACLE AVOIDANCE  (builds on Stage 1)")
    print("  2 obstacles ON the sine path at x=5 m and x=10 m")
    print("  Avoidance: always LEFT.  Min safe clearance: 0.43 m")
    print("  Target: Goals > 60%,  Coll = 0%")
    print("=" * 72)

    best_s1 = "models/stage1/best_model"
    if not os.path.exists(best_s1 + ".zip"):
        print("❌  Stage 1 model not found!")
        print("    Run: python train_robot.py --mode stage1  first.")
        return

    env      = Monitor(gym.make("SinusoidalRobot-v1-stage2", render_mode=None))
    eval_env = Monitor(gym.make("SinusoidalRobot-v1-stage2", render_mode=None))
    check_env(env)
    print("✓ Env OK")
    print(f"✓ Loading Stage 1 weights: {best_s1}.zip\n")

    os.makedirs("models/stage2", exist_ok=True)

    model = PPO.load(best_s1, env=env)
    model.ent_coef      = 0.01
    model.learning_rate = 2e-4

    model.learn(
        total_timesteps=timesteps,
        callback=[
            CheckpointCallback(
                save_freq=25_000, save_path="models/stage2/",
                name_prefix="robot_s2", verbose=0),
            EvalCallback(
                eval_env,
                best_model_save_path="models/stage2/",
                eval_freq=10_000, n_eval_episodes=20,
                deterministic=True, render=False, verbose=1),
            TrainingMonitor(stage=2),
        ],
        progress_bar=True,
        reset_num_timesteps=True)

    model.save("models/stage2/final")
    print("\n✅ STAGE 2 DONE")
    print("   models/stage2/best_model.zip")
    print("   Test:     python train_robot.py --mode test     --stage 2")
    print("   Showcase: python train_robot.py --mode showcase --stage 2\n")
    env.close()
    eval_env.close()


# ════════════════════════════════════════════════════════════════
# TEST
# ════════════════════════════════════════════════════════════════
def test(stage=2, episodes=10, model_path=None):
    env_id = f"SinusoidalRobot-v1-stage{stage}"
    if model_path is None:
        model_path = f"models/stage{stage}/best_model"
    if not os.path.exists(model_path + ".zip"):
        print(f"❌  Model not found: {model_path}.zip")
        return

    print(f"\nTesting Stage {stage}: {model_path}.zip")
    print("-" * 65)

    model = PPO.load(model_path)
    env   = gym.make(env_id, render_mode="human")
    goals, all_cte = 0, []

    for ep in range(episodes):
        obs, _ = env.reset()
        done, total_r, ep_cte = False, 0.0, []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += r
            ep_cte.append(abs(float(obs[0])))
            env.render()

        if info.get("goal_reached"): goals += 1
        all_cte.extend(ep_cte)
        n_obs = 2 if stage == 2 else 0
        print(f"  Ep {ep+1:2d}: R={total_r:7.0f} | "
              f"CTE={np.mean(ep_cte):.3f} m | "
              f"Passed={info.get('obstacles_passed',0)}/{n_obs} | "
              f"Goal={'YES ✓' if info.get('goal_reached') else 'NO  '}")

    print("-" * 65)
    print(f"  Goal rate: {goals/episodes*100:.0f}%  |  "
          f"Mean CTE: {np.mean(all_cte):.3f} m\n")
    env.close()


# ════════════════════════════════════════════════════════════════
# SHOWCASE — show only perfect episodes
# ════════════════════════════════════════════════════════════════
def showcase(stage=2, show_episodes=5, max_tries=200, model_path=None):
    env_id = f"SinusoidalRobot-v1-stage{stage}"
    if model_path is None:
        model_path = f"models/stage{stage}/best_model"
    if not os.path.exists(model_path + ".zip"):
        print(f"❌  Model not found: {model_path}.zip")
        return

    print("=" * 72)
    print(f"  SHOWCASE — PERFECT EPISODES ONLY  (Stage {stage})")
    print("  Perfect = goal + no collision + CTE < 0.15 m"
          + (" + both obstacles passed" if stage == 2 else ""))
    print("=" * 72)

    model      = PPO.load(model_path)
    silent_env = gym.make(env_id, render_mode=None)
    visual_env = gym.make(env_id, render_mode="human")
    shown = tried = 0

    while shown < show_episodes and tried < max_tries:
        tried += 1
        obs, _ = silent_env.reset()
        done, ep_cte, ep_info = False, [], {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, ep_info = silent_env.step(action)
            done = term or trunc
            ep_cte.append(abs(float(obs[0])))

        avg_cte = np.mean(ep_cte) if ep_cte else 999.0
        goal_ok = ep_info.get('goal_reached', False)
        coll_ok = not ep_info.get('collision', True)
        cte_ok  = avg_cte < 0.15
        pass_ok = ep_info.get('obstacles_passed', 0) >= 2 if stage == 2 else True
        perfect = goal_ok and coll_ok and cte_ok and pass_ok

        print(f"  Try {tried:3d} | "
              f"Goal:{'✓' if goal_ok else '✗'} | "
              f"Coll:{'✓' if coll_ok else '✗'} | "
              f"CTE:{avg_cte:.3f} {'✓' if cte_ok else '✗'} | "
              f"Pass:{'✓' if pass_ok else '✗'} | "
              f"{'→ SHOWING!' if perfect else 'skip'}")

        if perfect:
            shown += 1
            print(f"\n  ★ Perfect ep {shown}/{show_episodes} ...")
            obs, _ = visual_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = visual_env.step(action)
                done = term or trunc
                visual_env.render()
            print("  ✓ Done!\n")

    print(f"\n  Showed {shown}/{show_episodes} perfect eps in {tried} tries.")
    silent_env.close()
    visual_env.close()


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["stage1", "stage2", "test", "showcase"],
                   default="stage1")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--episodes",  type=int, default=10)
    p.add_argument("--stage",     type=int, default=2)
    p.add_argument("--max_tries", type=int, default=200)
    p.add_argument("--model",     default=None)
    args = p.parse_args()

    os.makedirs("models", exist_ok=True)

    if args.mode == "stage1":
        train_stage1(timesteps=args.timesteps or 1_500_000)
    elif args.mode == "stage2":
        train_stage2(timesteps=args.timesteps or 2_000_000)
    elif args.mode == "test":
        test(stage=args.stage, episodes=args.episodes, model_path=args.model)
    elif args.mode == "showcase":
        showcase(stage=args.stage, show_episodes=args.episodes,
                 max_tries=args.max_tries, model_path=args.model)
