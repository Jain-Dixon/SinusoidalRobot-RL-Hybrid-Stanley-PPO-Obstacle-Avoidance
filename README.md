# 🤖 Autonomous Differential Drive Robot

> PPO Reinforcement Learning • Sinusoidal Path Following • Obstacle Avoidance  
> Raspberry Pi 4 • Arduino Uno • RPLidar A1 • Cytron MDD10A

---

## 📌 Project Overview

A two-stage autonomous mobile robot that uses **Proximal Policy Optimization (PPO)**
to learn sinusoidal path following and real-time obstacle avoidance.
The trained policy is deployed on a physical differential-drive robot built on a
360 mm circular plywood chassis.

---

## 🏗️ Hardware

| Component | Specification |
|---|---|
| SBC | Raspberry Pi 4 (4 GB) |
| Microcontroller | Arduino Uno |
| Motor Driver | Cytron MDD10A (10 A dual channel) |
| Motors | Johnson 100 RPM DC gear motors |
| LiDAR | RPLidar A1 (360°, 5.5 Hz) |
| Camera | Raspberry Pi Camera Module |
| Encoders | LM393 optical (10-slot disc) |
| IMU | MPU9250 (I²C, 3.3 V) |
| Battery | 3S LiPo |
| Chassis | 360 mm circular plywood, 2-deck |

---

## 🧠 Reinforcement Learning Pipeline

### Stage 1 — Sine Path Following
- Environment: `SinusoidalRobotEnv` (stage=1)
- Observation space: 26-dim vector (CTE, heading error, goal distance, velocity, 16-ray LiDAR, 5 look-ahead cues)
- Action space: [linear speed, angular rate]
- Target: Goal rate > 80%, avg CTE < 0.10 m

### Stage 2 — Obstacle Avoidance
- Fine-tuned from Stage 1 weights
- 2 obstacles placed directly on the sine path (x = 5 m, x = 10 m)
- Robot must detect → steer LEFT → clear → snap back → continue
- Target: Goal rate > 60%, Collision = 0%

### Reward Design
| Condition | Reward |
|---|---|
| On path (CTE ≤ 0.10 m) | +1.5 / step |
| Forward progress | +1.0 × (dx/max_dx) |
| Obstacle cleared | +30 |
| Goal reached (on path) | +500 |
| Collision | −100 |
| Out of bounds | −50 |

---

## 📁 Repository Structure
```
├── simulation/
│   ├── robot_env.py        # Gymnasium environment
│   └── train_robot.py      # PPO training + evaluation
├── hardware/
│   ├── raspberry_pi/       # Pi deployment controller
│   └── arduino/            # Encoder PID firmware
├── models/                 # Saved model checkpoints
├── docs/                   # Wiring diagrams
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Stage 1 (path following)
```bash
python simulation/train_robot.py --mode stage1
```

### 3. Train Stage 2 (obstacle avoidance)
```bash
python simulation/train_robot.py --mode stage2
```

### 4. Test visually
```bash
python simulation/train_robot.py --mode test --stage 2
```

### 5. Showcase perfect episodes only
```bash
python simulation/train_robot.py --mode showcase --stage 2
```

---

## 📊 Training Results

| Stage | Goal Rate | Avg CTE | Collisions |
|---|---|---|---|
| Stage 1 | ~85% | ~0.08 m | — |
| Stage 2 | ~70% | ~0.11 m | ~0% |

---

## 📐 Wiring Overview

- **Pi ↔ Arduino**: USB Serial
- **Arduino ↔ Cytron MDD10A**: PWM + DIR pins
- **IMU (MPU9250)**: I²C on Pi (SDA=GPIO2, SCL=GPIO3)
- **LiDAR (RPLidar A1)**: USB on Pi
- **Encoders (LM393)**: Digital interrupt pins on Arduino
- **Power**: 3S LiPo → XL4015 buck → 5V for Pi/Arduino; direct for motors

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Author

Built as part of an academic robotics project.  
Feel free to open an issue or pull request!
