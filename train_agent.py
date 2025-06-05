#!/usr/bin/env python
"""Train Q-learning agent with live visualization.

This script shows the agent learning in real-time by rendering episodes
during training. You can watch how it fails initially and gradually improves.
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def train_with_visualization(
    map_name: str = "4x4",
    is_slippery: bool = True,
    episodes: int = 10_000,
    alpha: float = 0.8,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.9995,
    min_epsilon: float = 0.01,
    render_interval: int = 1000,
    render_episodes: int = 3,
):
    """Train Q-learning with periodic visualization.
    
    Parameters
    ----------
    render_interval : int
        Show agent playing every N episodes
    render_episodes : int  
        Number of episodes to render each time
    """
    
    # Training environment (no rendering for speed)
    train_env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    
    # Visualization environment
    render_env = gym.make(
        "FrozenLake-v1", 
        map_name=map_name, 
        is_slippery=is_slippery, 
        render_mode="human"
    )
    
    state_space = train_env.observation_space.n
    action_space = train_env.action_space.n
    q_table = np.zeros((state_space, action_space), dtype=np.float32)
    
    episode_rewards = []
    
    print(f"Training on {map_name} map (slippery={is_slippery})")
    print(f"Will show visualization every {render_interval} episodes\n")
    
    for ep in range(episodes):
        state, _ = train_env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = train_env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
            
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            
            # Q-learning update
            best_next = float(np.max(q_table[next_state]))
            td_target = reward + gamma * best_next
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error
            
            total_reward += reward
            state = next_state
        
        # Decay exploration
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        
        # Show progress and visualization
        if (ep + 1) % render_interval == 0:
            avg_reward = np.mean(episode_rewards[-render_interval:])
            print(f"\n=== Episode {ep + 1:,} ===")
            print(f"Average reward (last {render_interval}): {avg_reward:.3f}")
            print(f"Exploration rate (ε): {epsilon:.3f}")
            print(f"Q-table sparsity: {np.count_nonzero(q_table)}/{q_table.size} entries learned")
            
            print(f"\nShowing agent performance (playing {render_episodes} episodes):")
            
            # Show agent playing
            for demo_ep in range(render_episodes):
                state, _ = render_env.reset()
                done = False
                steps = 0
                demo_reward = 0.0
                
                print(f"  Demo episode {demo_ep + 1}/{render_episodes}")
                
                while not done and steps < 100:  # Prevent infinite loops
                    # Use learned policy (greedy)
                    action = int(np.argmax(q_table[state]))
                    state, reward, terminated, truncated, _ = render_env.step(action)
                    done = terminated or truncated
                    demo_reward += reward
                    steps += 1
                    time.sleep(0.3)  # Slow down for human observation
                
                result = "SUCCESS! 🎉" if demo_reward > 0 else "Failed ❌"
                print(f"    → {result} (reward: {demo_reward}, steps: {steps})")
            
            print("-" * 50)
    
    train_env.close()
    render_env.close()
    
    return q_table, episode_rewards


def plot_training_progress(rewards, save_path=None):
    """Plot training progress with multiple metrics."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Running average
    window = min(100, len(rewards) // 10)
    if window > 0:
        running_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax1.plot(running_avg, 'b-', linewidth=2)
        ax1.set_title(f'Running Average Reward (window={window})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        ax1.grid(True, alpha=0.3)
    
    # Success rate over time (for binary rewards)
    batch_size = max(100, len(rewards) // 50)
    success_rates = []
    batch_centers = []
    
    for i in range(0, len(rewards), batch_size):
        batch = rewards[i:i+batch_size]
        success_rate = np.mean(batch)
        success_rates.append(success_rate)
        batch_centers.append(i + batch_size//2)
    
    ax2.plot(batch_centers, success_rates, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_title(f'Success Rate Over Time (batches of {batch_size})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training progress saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train Q-learning agent with visualization")
    parser.add_argument("--map", choices=["4x4", "8x8"], default="4x4", help="Map size")
    parser.add_argument("--deterministic", action="store_true", help="Disable slippery tiles")
    parser.add_argument("--episodes", type=int, default=10_000, help="Training episodes")
    parser.add_argument("--alpha", type=float, default=0.8, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Exploration decay")
    parser.add_argument("--min-epsilon", type=float, default=0.01, help="Minimum exploration")
    parser.add_argument("--render-interval", type=int, default=1000, help="Show demo every N episodes")
    parser.add_argument("--render-episodes", type=int, default=3, help="Episodes to show each demo")
    parser.add_argument("--save-table", default="q_table.npy", help="Where to save learned Q-table")
    parser.add_argument("--no-plot", action="store_true", help="Skip progress plot")
    
    args = parser.parse_args()
    
    is_slippery = not args.deterministic
    
    print("🚀 Starting Q-learning training with live visualization!")
    print("💡 Watch how the agent learns from failures to success...\n")
    
    q_table, rewards = train_with_visualization(
        map_name=args.map,
        is_slippery=is_slippery,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        render_interval=args.render_interval,
        render_episodes=args.render_episodes,
    )
    
    # Save Q-table
    np.save(args.save_table, q_table)
    print(f"\n✅ Training complete! Q-table saved to {args.save_table}")
    
    # Plot progress
    if not args.no_plot:
        plot_path = Path(__file__).with_suffix(".png")
        plot_training_progress(rewards, save_path=plot_path)
    
    # Final statistics
    recent_success = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    print(f"📊 Final success rate: {recent_success*100:.1f}%")
    print(f"📈 Total episodes trained: {len(rewards):,}")


if __name__ == "__main__":
    main() 