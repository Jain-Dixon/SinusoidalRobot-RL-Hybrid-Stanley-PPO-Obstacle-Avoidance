"""
robot_env.py  —  Sinusoidal Robot RL Environment
=================================================

Stage 1 — Path following only (no obstacles)
Stage 2 — Obstacle avoidance + return to path

Reward scale design
-------------------
Per-step rewards are small so episode totals are in a readable range:
  Good episode  (on path, reaches goal) :  ~2000–3500
  Bad  episode  (off path constantly)   :  ~-3000 to -8000
  Collision / boundary                  :  -100 / -50  (not overwhelming)
This prevents the agent from "panicking" due to huge per-step penalties.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class SinusoidalRobotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}

    def __init__(self, render_mode=None, stage=1):
        super().__init__()

        self.stage = stage

        # ── Robot ────────────────────────────────────────────────
        self.robot_radius  = 0.18
        self.max_speed     = 0.4         # m/s
        self.max_angular   = 0.6         # rad/s
        self.dt            = 0.05        # s per step
        self.max_steps     = 2000

        # ── Sine path ────────────────────────────────────────────
        self.path_amplitude = 0.50
        self.path_frequency = 0.15

        # ── Goal ─────────────────────────────────────────────────
        self.goal_x         = 15.0
        self.goal_threshold = 0.50

        # ── Obstacles ────────────────────────────────────────────
        self.num_obstacles        = 2
        self.obs_radius           = 0.25
        self.min_safe_lateral     = self.robot_radius + self.obs_radius  # 0.43 m
        self.obs_clear_dist       = self.obs_radius + 0.30               # 0.55 m

        # ── Spaces ───────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32)

        # ── Internal state ───────────────────────────────────────
        self.state              = None
        self.obstacles          = []
        self.trajectory         = []
        self.steps              = 0
        self.goal_reached       = False
        self.obstacles_passed   = set()
        self.currently_avoiding = False
        self._cached_lidar      = None
        self._prev_passed_count = 0

        self.render_mode = render_mode
        self.fig = None
        self.ax  = None

    # ════════════════════════════════════════════════════════════
    # PATH
    # ════════════════════════════════════════════════════════════
    def _path_y(self, x):
        return self.path_amplitude * np.sin(2 * np.pi * self.path_frequency * x)

    def _path_heading(self, x):
        dy = (self.path_amplitude * 2 * np.pi * self.path_frequency
              * np.cos(2 * np.pi * self.path_frequency * x))
        return float(np.arctan2(dy, 1.0))

    # ════════════════════════════════════════════════════════════
    # RESET
    # ════════════════════════════════════════════════════════════
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = {
            'x': 0.0, 'y': 0.0,
            'theta': self._path_heading(0.0),
            'vx': 0.0, 'vy': 0.0,
        }
        self.steps              = 0
        self.goal_reached       = False
        self.currently_avoiding = False
        self.obstacles_passed   = set()
        self._prev_passed_count = 0
        self.trajectory         = [(0.0, 0.0)]
        self._cached_lidar      = None
        self.obstacles = self._make_obstacles() if self.stage == 2 else []
        return self._obs(), {}

    # ════════════════════════════════════════════════════════════
    # OBSTACLES — exactly on the sine path
    # ════════════════════════════════════════════════════════════
    def _make_obstacles(self):
        return [
            {'id': i, 'x': float(x), 'y': float(self._path_y(x)),
             'radius': self.obs_radius}
            for i, x in enumerate([5.0, 10.0])
        ]

    # ════════════════════════════════════════════════════════════
    # LIDAR — 16 rays, 3 m range, normalised [0,1]
    # ════════════════════════════════════════════════════════════
    def _compute_lidar(self):
        R  = 3.0
        rx, ry, th = self.state['x'], self.state['y'], self.state['theta']
        rays = np.full(16, R, dtype=np.float64)
        for i in range(16):
            ang    = th + 2 * np.pi * i / 16
            ca, sa = np.cos(ang), np.sin(ang)
            for o in self.obstacles:
                ex, ey = o['x'] - rx, o['y'] - ry
                t      = ex * ca + ey * sa
                if t <= 0:
                    continue
                perp = np.hypot(ex - t * ca, ey - t * sa)
                if perp < o['radius']:
                    hit = t - np.sqrt(max(0.0, o['radius'] ** 2 - perp ** 2))
                    rays[i] = min(rays[i], max(0.0, hit))
        return rays / R

    def _lidar(self):
        if self._cached_lidar is None:
            self._cached_lidar = self._compute_lidar()
        return self._cached_lidar

    # ════════════════════════════════════════════════════════════
    # OBSERVATION (26,)
    # [0]    cte_signed  y - path_y   (+= LEFT of path)
    # [1]    h_err       heading error
    # [2]    g_dist      (goal_x-x)/goal_x  [0,1]
    # [3]    vx_n        vx/max_speed
    # [4]    vy_n        vy/max_speed
    # [5-20] lidar       16 rays [0,1]
    # [21-25]ahead       5 lookahead cues [-1,1]
    # ════════════════════════════════════════════════════════════
    def _obs(self):
        x, y, th = self.state['x'], self.state['y'], self.state['theta']
        path_y = self._path_y(x)
        cte    = float(y - path_y)
        ideal  = self._path_heading(x)
        h_err  = float(np.arctan2(np.sin(ideal - th), np.cos(ideal - th)))
        g_dist = float(np.clip((self.goal_x - x) / self.goal_x, 0.0, 1.0))
        vx_n   = float(self.state['vx'] / self.max_speed)
        vy_n   = float(self.state['vy'] / self.max_speed)
        lidar  = self._lidar()
        ahead  = [
            float(np.clip(
                (self._path_y(x + d) - y) / self.path_amplitude, -1.0, 1.0))
            for d in [0.5, 1.0, 1.5, 2.0, 2.5]
        ]
        return np.array(
            [cte, h_err, g_dist, vx_n, vy_n, *lidar, *ahead],
            dtype=np.float32)

    # ════════════════════════════════════════════════════════════
    # STEP
    # ════════════════════════════════════════════════════════════
    def step(self, action):
        self._cached_lidar = None
        prev_x = self.state['x']

        # Minimum forward speed — cannot reverse
        lin = max(0.15, float(np.clip(action[0], 0.0, 1.0)) * self.max_speed)
        ang = float(np.clip(action[1], -1.0, 1.0)) * self.max_angular

        self.state['theta'] += ang * self.dt
        self.state['theta']  = float(np.arctan2(
            np.sin(self.state['theta']),
            np.cos(self.state['theta'])))

        self.state['x']  += lin * np.cos(self.state['theta']) * self.dt
        self.state['y']  += lin * np.sin(self.state['theta']) * self.dt
        self.state['vx']  = lin * np.cos(self.state['theta'])
        self.state['vy']  = lin * np.sin(self.state['theta'])

        self.trajectory.append((self.state['x'], self.state['y']))
        if len(self.trajectory) > 3000:
            self.trajectory = self.trajectory[-3000:]

        # Obstacle passed threshold — matches obs_ahead window end exactly
        for o in self.obstacles:
            if o['id'] not in self.obstacles_passed:
                if self.state['x'] > o['x'] + self.obs_clear_dist:
                    self.obstacles_passed.add(o['id'])

        # Goal
        if np.hypot(self.state['x'] - self.goal_x,
                    self.state['y'] - self._path_y(self.goal_x)) < self.goal_threshold:
            self.goal_reached = True

        # Collision
        collision = any(
            np.hypot(self.state['x'] - o['x'], self.state['y'] - o['y'])
            < self.robot_radius + o['radius']
            for o in self.obstacles)

        reward     = self._reward(prev_x, collision)
        self.steps += 1

        terminated = bool(
            self.goal_reached or collision
            or abs(self.state['y']) > 2.5
            or self.steps >= self.max_steps)

        boundary_exit = abs(self.state['y']) > 2.5

        return self._obs(), reward, terminated, False, {
            'goal_reached':     self.goal_reached,
            'collision':        collision,
            'boundary_exit':    boundary_exit,
            'obstacles_passed': len(self.obstacles_passed),
        }

    # ════════════════════════════════════════════════════════════
    # REWARD
    # ════════════════════════════════════════════════════════════
    def _reward(self, prev_x, collision):
        x, y, th = self.state['x'], self.state['y'], self.state['theta']
        path_y   = self._path_y(x)
        cte      = abs(y - path_y)
        cte_sign = y - path_y           # + = LEFT of path
        lidar    = self._lidar()
        min_l    = float(np.min(lidar))

        # ── Hard penalties ────────────────────────────────────────
        if collision:
            return -100.0
        if abs(y) > 2.5:
            return -50.0

        # ── Avoidance detection ───────────────────────────────────
        lidar_threat = (min_l < 0.50) and (self.stage == 2)

        obs_ahead = False
        for o in self.obstacles:
            if o['id'] in self.obstacles_passed:
                continue
            # <= so window closes at exact same step obstacle is marked passed
            if (o['x'] - 3.0 < x) and (x <= o['x'] + self.obs_clear_dist):
                obs_ahead = True
                break

        self.currently_avoiding = lidar_threat or obs_ahead

        r = 0.0

        # ════════════════════════════════════════════════════════
        # RULE 1 — FOLLOW SINE PATH
        # ════════════════════════════════════════════════════════
        if not self.currently_avoiding:

            # Cross-track error  (small per-step values)
            if cte <= 0.10:
                r += 1.5
            elif cte <= 0.25:
                r += 0.5
            elif cte <= 0.50:
                r -= 1.5
            else:
                r -= 5.0       # bad but not catastrophic

            # Heading alignment
            ideal = self._path_heading(x)
            h_err = abs(np.arctan2(np.sin(ideal - th), np.cos(ideal - th)))
            if h_err < 0.15:
                r += 0.5
            elif h_err < 0.35:
                r += 0.1
            else:
                r -= 1.5 * h_err

        # ════════════════════════════════════════════════════════
        # RULE 2 — AVOID OBSTACLE (stage 2)
        # Go LEFT with proportional reward gradient.
        # Safe clearance = 0.43 m, target = 0.55 m.
        # ════════════════════════════════════════════════════════
        else:
            # (a) Proportional left-side reward
            #     Smooth gradient: 0 at cte_sign=0, +3 at cte_sign=0.55+
            if cte_sign < 0:
                r -= 3.0    # wrong side
            else:
                r += min(3.0, (cte_sign / 0.55) * 3.0)

            # Soft cap at 0.90 m left
            if cte_sign > 0.90:
                r -= 2.0 * (cte_sign - 0.90)

            # (b) LIDAR clearance
            if min_l > 0.45:
                r += 1.5
            elif min_l > 0.30:
                r += 0.5
            elif min_l > 0.15:
                r += 0.1
            else:
                r -= 4.0    # dangerously close

            # (c) Penalise backward heading
            if np.cos(th) < 0:
                r -= 2.0

        # ════════════════════════════════════════════════════════
        # RULE 3 — ALWAYS MOVE FORWARD
        # ════════════════════════════════════════════════════════
        dx = self.state['x'] - prev_x
        if dx > 0.001:
            r += 1.0 * (dx / (self.max_speed * self.dt))
        else:
            r -= 2.0    # stall penalty

        # ════════════════════════════════════════════════════════
        # RULE 4 — SNAP BACK bonus when obstacle cleared
        # ════════════════════════════════════════════════════════
        newly = len(self.obstacles_passed) - self._prev_passed_count
        self._prev_passed_count = len(self.obstacles_passed)

        if newly > 0:
            r += 30.0 * newly
            if cte <= 0.20:
                r += 15.0
            elif cte <= 0.40:
                r += 5.0

        # Goal bonus
        if self.goal_reached:
            if cte <= 0.10:
                r += 500.0
            elif cte <= 0.30:
                r += 200.0
            else:
                r += 50.0

        return float(r)

    # ════════════════════════════════════════════════════════════
    # RENDER
    # ════════════════════════════════════════════════════════════
    def render(self):
        if self.render_mode is None:
            return None
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(18, 6))

        ax = self.ax
        ax.clear()
        ax.set_facecolor('#eef6ff')

        xs = np.linspace(0, self.goal_x + 0.5, 800)
        ys = np.array([self._path_y(xi) for xi in xs])

        ax.fill_between(xs, ys - 0.10, ys + 0.10, alpha=0.60,
                        color='limegreen', label='±10 cm')
        ax.fill_between(xs, ys - 0.30, ys + 0.30, alpha=0.25,
                        color='yellow',   label='±30 cm')
        ax.fill_between(xs, ys - 0.60, ys + 0.60, alpha=0.10,
                        color='orange',   label='±60 cm')
        ax.plot(xs, ys, 'g-', linewidth=3, label='Sine path', zorder=4)

        for o in self.obstacles:
            done = o['id'] in self.obstacles_passed
            ax.add_patch(Circle(
                (o['x'], o['y']), o['radius'],
                color='gray' if done else 'tomato',
                alpha=0.30   if done else 0.85,
                zorder=5, edgecolor='darkred', lw=2))
            if done:
                ax.text(o['x'], o['y'], '✓', ha='center', va='center',
                        fontsize=16, color='green', fontweight='bold')
            else:
                ax.text(o['x'], o['y'] + o['radius'] + 0.14,
                        f"OBS {o['id']+1}", ha='center', fontsize=9,
                        color='darkred', fontweight='bold')

        rx, ry, th = self.state['x'], self.state['y'], self.state['theta']
        lidar = self._lidar()
        if min(lidar) < 0.65:
            for i in range(16):
                ang = th + 2 * np.pi * i / 16
                d   = lidar[i] * 3.0
                col = ('red'    if lidar[i] < 0.15 else
                       'orange' if lidar[i] < 0.35 else 'steelblue')
                ax.plot([rx, rx + d*np.cos(ang)], [ry, ry + d*np.sin(ang)],
                        color=col, alpha=0.55, lw=1.5, zorder=3)

        if len(self.trajectory) > 1:
            t = np.array(self.trajectory)
            ax.plot(t[:, 0], t[:, 1], color='royalblue', lw=2.5,
                    alpha=0.85, label='Robot path', zorder=6)

        body_col = ('gold'        if self.goal_reached
                    else 'darkorange' if self.currently_avoiding
                    else 'dodgerblue')
        ax.add_patch(Circle((rx, ry), self.robot_radius,
                             color=body_col, zorder=10,
                             edgecolor='black', lw=2))
        ax.annotate('',
                    xy=(rx + 0.4*np.cos(th), ry + 0.4*np.sin(th)),
                    xytext=(rx, ry),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2.5),
                    zorder=11)

        ax.plot(0, 0, 'o', markersize=14, color='limegreen', zorder=8,
                markeredgecolor='darkgreen', mew=2, label='Start')
        ax.plot(self.goal_x, self._path_y(self.goal_x), '*', markersize=30,
                color='gold', zorder=8, markeredgecolor='darkorange',
                mew=2, label='Goal')

        cte  = abs(self.state['y'] - self._path_y(rx))
        scol = ('green' if cte < 0.10 else 'darkorange' if cte < 0.30 else 'red')
        mode = ('🎯 GOAL!'             if self.goal_reached
                else '⚠ AVOIDING (LEFT)' if self.currently_avoiding
                else '✓ ON PATH')
        stag = ('STAGE 1 — PATH ONLY' if self.stage == 1
                else 'STAGE 2 — PATH + 2 OBSTACLES  [go LEFT]')

        ax.set_xlim(-0.5, self.goal_x + 1.0)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, ls='--')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'{stag}  |  Step {self.steps}/{self.max_steps}  |  '
            f'CTE: {cte:.3f} m  |  '
            f'Passed: {len(self.obstacles_passed)}/{self.num_obstacles}  |  {mode}',
            fontsize=12, fontweight='bold', color=scol)
        plt.tight_layout()
        plt.pause(0.001)
        return None

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


# ════════════════════════════════════════════════════════════════
# GYM REGISTRATION
# ════════════════════════════════════════════════════════════════
gym.register(id='SinusoidalRobot-v1-stage1',
             entry_point='robot_env:SinusoidalRobotEnv',
             kwargs={'stage': 1})

gym.register(id='SinusoidalRobot-v1-stage2',
             entry_point='robot_env:SinusoidalRobotEnv',
             kwargs={'stage': 2})
