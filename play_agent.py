#!/usr/bin/env python
"""Play with a trained Q-learning agent.

Load a saved Q-table and watch the agent play optimally.
Perfect for quick demos during presentations!
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np


def play_trained_agent(
    q_table,
    map_name: str = "4x4",
    is_slippery: bool = True,
    episodes: int = 5,
    delay: float = 0.8,
):
    """Let trained agent play multiple episodes."""
    
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode="human")
    
    print(f"🎮 Trained agent playing on {map_name} map (slippery={is_slippery})")
    print(f"⏱️  Playing {episodes} episodes with {delay}s delay between moves\n")
    
    total_successes = 0
    total_steps = 0
    
    for ep in range(episodes):
        print(f"--- Episode {ep + 1}/{episodes} ---")
        
        state, _ = env.reset()
        done = False
        steps = 0
        episode_reward = 0.0
        
        while not done and steps < 200:  # Prevent infinite loops
            # Use trained policy (greedy)
            action = int(np.argmax(q_table[state]))
            
            # Action mapping for display
            actions = ["⬅️ Left", "⬇️ Down", "➡️ Right", "⬆️ Up"]
            print(f"  Step {steps + 1}: {actions[action]}")
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            time.sleep(delay)
        
        # Episode results
        success = episode_reward > 0
        total_successes += int(success)
        total_steps += steps
        
        result = "SUCCESS! 🎉" if success else "Failed ❌"
        print(f"  → {result} (reward: {episode_reward}, steps: {steps})")
        
        if ep < episodes - 1:
            print("\nPress Enter for next episode...")
            input()
        
        print()
    
    env.close()
    
    # Summary
    success_rate = total_successes / episodes
    avg_steps = total_steps / episodes
    
    print("=" * 40)
    print(f"📊 SUMMARY:")
    print(f"   Success rate: {success_rate*100:.1f}% ({total_successes}/{episodes})")
    print(f"   Average steps: {avg_steps:.1f}")
    print("=" * 40)


def analyze_q_table(q_table, map_name: str = "4x4"):
    """Show some insights about the learned Q-table."""
    
    print(f"\n🧠 Q-TABLE ANALYSIS:")
    print(f"   Shape: {q_table.shape}")
    print(f"   Total entries: {q_table.size}")
    print(f"   Non-zero entries: {np.count_nonzero(q_table)} ({np.count_nonzero(q_table)/q_table.size*100:.1f}%)")
    print(f"   Max Q-value: {np.max(q_table):.3f}")
    print(f"   Min Q-value: {np.min(q_table):.3f}")
    
    # Show learned policy (best action for each state)
    policy = np.argmax(q_table, axis=1)
    actions = ["←", "↓", "→", "↑"]
    
    print(f"\n📋 LEARNED POLICY (best action for each state):")
    
    if map_name == "4x4":
        size = 4
    else:  # 8x8
        size = 8
    
    for row in range(size):
        policy_row = ""
        for col in range(size):
            state = row * size + col
            if state < len(policy):
                policy_row += f" {actions[policy[state]]} "
            else:
                policy_row += " ? "
        print(f"   {policy_row}")


def main():
    parser = argparse.ArgumentParser(description="Play with trained Q-learning agent")
    parser.add_argument("--q-table", default="q_table.npy", help="Path to trained Q-table")
    parser.add_argument("--map", choices=["4x4", "8x8"], default="4x4", help="Map size")
    parser.add_argument("--deterministic", action="store_true", help="Disable slippery tiles")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay between moves (seconds)")
    parser.add_argument("--analyze", action="store_true", help="Show Q-table analysis")
    parser.add_argument("--continuous", action="store_true", help="Play continuously without waiting for Enter")
    
    args = parser.parse_args()
    
    # Load Q-table
    q_table_path = Path(args.q_table)
    if not q_table_path.exists():
        print(f"❌ Error: Q-table file '{args.q_table}' not found!")
        print("💡 Available options:")
        print("   1. Run 'python train_agent.py' to train a new agent")
        print("   2. Specify different file with --q-table <filename>")
        return
    
    try:
        q_table = np.load(q_table_path)
        print(f"✅ Loaded Q-table from {args.q_table}")
    except Exception as e:
        print(f"❌ Error loading Q-table: {e}")
        return
    
    if args.analyze:
        analyze_q_table(q_table, args.map)
    
    is_slippery = not args.deterministic
    
    # Override input() for continuous play
    if args.continuous:
        global input
        input = lambda x="": None
    
    play_trained_agent(
        q_table=q_table,
        map_name=args.map,
        is_slippery=is_slippery,
        episodes=args.episodes,
        delay=args.delay,
    )


if __name__ == "__main__":
    main() 