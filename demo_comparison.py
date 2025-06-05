#!/usr/bin/env python
"""Demo comparison: Random Agent vs Trained Agent.

This script loads a trained Q-table and shows the dramatic difference
between random actions and learned policy.
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np


def evaluate_agent(
    q_table,
    map_name: str = "4x4", 
    is_slippery: bool = True,
    episodes: int = 10,
    agent_name: str = "Agent",
):
    """Evaluate agent performance over multiple episodes."""
    
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    
    successes = 0
    total_steps = 0
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:  # Prevent infinite loops
            if q_table is None:
                # Random agent
                action = env.action_space.sample()
            else:
                # Trained agent
                action = int(np.argmax(q_table[state]))
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if done and reward > 0:
                successes += 1
                break
        
        total_steps += steps
    
    env.close()
    
    success_rate = successes / episodes
    avg_steps = total_steps / episodes
    
    return success_rate, avg_steps


def demo_side_by_side(
    q_table,
    map_name: str = "4x4",
    is_slippery: bool = True,
    episodes: int = 5,
):
    """Show random vs trained agent playing side by side."""
    
    print(f"\n🎮 DEMO: Random vs Trained Agent")
    print(f"Map: {map_name}, Slippery: {is_slippery}")
    print("=" * 60)
    
    for ep in range(episodes):
        print(f"\n--- Episode {ep + 1}/{episodes} ---")
        
        # Random agent
        print("🎲 Random Agent:")
        random_result = play_single_episode(None, map_name, is_slippery, render=True)
        
        print("\n🧠 Trained Agent:")
        trained_result = play_single_episode(q_table, map_name, is_slippery, render=True)
        
        print(f"\nResults:")
        print(f"  Random:  {'SUCCESS! 🎉' if random_result['success'] else 'Failed ❌'} ({random_result['steps']} steps)")
        print(f"  Trained: {'SUCCESS! 🎉' if trained_result['success'] else 'Failed ❌'} ({trained_result['steps']} steps)")
        
        if ep < episodes - 1:
            print("\nPress Enter for next episode...")
            input()


def play_single_episode(q_table, map_name, is_slippery, render=False):
    """Play one episode and return results."""
    
    if render:
        env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode="human")
    else:
        env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    
    state, _ = env.reset()
    done = False
    steps = 0
    success = False
    
    while not done and steps < 200:
        if q_table is None:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
        
        if render:
            time.sleep(0.5)
        
        if done and reward > 0:
            success = True
    
    env.close()
    
    return {
        "success": success,
        "steps": steps,
        "reward": 1.0 if success else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="Compare random vs trained agent")
    parser.add_argument("--q-table", required=True, help="Path to trained Q-table (.npy file)")
    parser.add_argument("--map", choices=["4x4", "8x8"], default="4x4", help="Map size")
    parser.add_argument("--deterministic", action="store_true", help="Disable slippery tiles") 
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes for statistical evaluation")
    parser.add_argument("--demo-episodes", type=int, default=5, help="Episodes for visual demonstration")
    parser.add_argument("--stats-only", action="store_true", help="Only show statistics, skip visual demo")
    
    args = parser.parse_args()
    
    # Load Q-table
    q_table_path = Path(args.q_table)
    if not q_table_path.exists():
        print(f"❌ Error: Q-table file '{args.q_table}' not found!")
        print("💡 Run 'python train_agent.py' first to create a trained agent.")
        return
    
    try:
        q_table = np.load(q_table_path)
        print(f"✅ Loaded Q-table from {args.q_table}")
        print(f"   Shape: {q_table.shape}, Non-zero entries: {np.count_nonzero(q_table)}")
    except Exception as e:
        print(f"❌ Error loading Q-table: {e}")
        return
    
    is_slippery = not args.deterministic
    
    # Statistical evaluation
    print(f"\n📊 STATISTICAL EVALUATION ({args.eval_episodes} episodes)")
    print("=" * 60)
    
    random_success, random_steps = evaluate_agent(
        None, args.map, is_slippery, args.eval_episodes, "Random"
    )
    
    trained_success, trained_steps = evaluate_agent(
        q_table, args.map, is_slippery, args.eval_episodes, "Trained"
    )
    
    print(f"Random Agent:  {random_success*100:5.1f}% success rate, {random_steps:5.1f} avg steps")
    print(f"Trained Agent: {trained_success*100:5.1f}% success rate, {trained_steps:5.1f} avg steps")
    
    improvement = (trained_success - random_success) / max(random_success, 0.001) * 100
    print(f"\n🚀 Improvement: {improvement:+.1f}% relative success rate increase!")
    
    # Visual demonstration
    if not args.stats_only:
        demo_side_by_side(q_table, args.map, is_slippery, args.demo_episodes)
    
    print(f"\n✨ Demo complete! The trained agent clearly outperforms random actions.")


if __name__ == "__main__":
    main() 