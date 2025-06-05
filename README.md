# Q-learning Demo Suite (FrozenLake)

This project contains multiple scripts to demonstrate **Q-learning** reinforcement learning on the classic `FrozenLake-v1` environment. Each script serves a specific purpose for your presentation needs.

## 🗂️ Files Overview

| Script | Purpose | Key Feature |
|--------|---------|-------------|
| `train_agent.py` | Train with live visualization | Watch agent learn in real-time |
| `demo_comparison.py` | Compare random vs trained | Side-by-side before/after demo |
| `play_agent.py` | Load & play with trained agent | Quick demo without retraining |
| `q_learning_demo.py` | Original all-in-one script | Complete training + evaluation |

## 1. Setup

1. Ensure you have **Python ≥ 3.10** installed.
2. Install the dependencies inside a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On the first run Gymnasium will ask to install some extra packages for rendering. Simply follow the prompt or add them manually if needed.

## 2. Quick Start Guide

### Step 1: Train an Agent (with visualization)

```bash
# Train on 4×4 map, show progress every 1000 episodes
python train_agent.py

# Train on harder 8×8 map with more episodes
python train_agent.py --map 8x8 --episodes 20000

# Train on deterministic (non-slippery) environment
python train_agent.py --deterministic
```

This will:
- Show live visualization every 1000 episodes during training
- Save the trained Q-table to `q_table.npy`
- Generate training progress plots

### Step 2: Compare Performance

```bash
# Compare random vs trained agent (requires trained Q-table)
python demo_comparison.py --q-table q_table.npy

# Quick stats-only comparison
python demo_comparison.py --q-table q_table.npy --stats-only
```

### Step 3: Play with Trained Agent

```bash
# Load and watch trained agent play
python play_agent.py --q-table q_table.npy

# Analyze Q-table and show learned policy
python play_agent.py --q-table q_table.npy --analyze

# Continuous play (no waiting between episodes)
python play_agent.py --q-table q_table.npy --continuous
```

## 3. Advanced Usage

### Training Options

```bash
# Fine-tune hyperparameters
python train_agent.py --alpha 0.6 --gamma 0.99 --epsilon-decay 0.999

# Show visualization more/less frequently
python train_agent.py --render-interval 500  # every 500 episodes
python train_agent.py --render-episodes 5    # show 5 demos each time

# Save to custom file
python train_agent.py --save-table my_agent.npy
```

### Demo Presentation Tips

**For live presentations:**
1. Pre-train agents: `python train_agent.py --map 4x4` và `python train_agent.py --map 8x8`
2. Quick comparison: `python demo_comparison.py --q-table q_table.npy --demo-episodes 3`
3. Show policy: `python play_agent.py --analyze --episodes 3`

**For different difficulties:**
- Easy: `--map 4x4 --deterministic` (deterministic 4×4)
- Medium: `--map 4x4` (stochastic 4×4, default)  
- Hard: `--map 8x8` (stochastic 8×8)

## 4. Talking Points for Presentation

*  **Reinforcement Learning loop**: state → action → reward → next state.
*  **Tabular Q-learning update rule**:  
   $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
*  **Exploration vs. exploitation**: ε-greedy strategy with exponential decay.
*  **Learning visualization**: Watch agent fail initially, then gradually improve.
*  **Performance comparison**: Dramatic difference between random (6%) vs trained (90%+) success rates.
*  **Policy interpretation**: Use `--analyze` flag to show learned optimal actions for each grid cell.

## 5. Legacy Script

The original `q_learning_demo.py` is still available and combines training + evaluation in one script:

```bash
python q_learning_demo.py --map 8x8 --episodes 15000
``` 