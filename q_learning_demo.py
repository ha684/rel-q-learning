#!/usr/bin/env python
"""Simple Q-learning demo using FrozenLake-v1 environment.

Run this script to train an agent with Q-learning and then watch it play.

Usage:
    python q_learning_demo.py           # trains and plays once finished

Adapt hyper-parameters with CLI flags:
    python q_learning_demo.py --episodes 15000 --alpha 0.7 --epsilon 0.9
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# -------------------------- Q-learning core -------------------------- #

def q_learning(
    env_name: str = "FrozenLake-v1",
    map_name: str = "4x4",
    is_slippery: bool = True,
    episodes: int = 10_000,
    alpha: float = 0.8,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.9995,
    min_epsilon: float = 0.01,
):
    """Train a tabular Q-learning agent.

    Returns
    -------
    q_table : np.ndarray
        Learned action-value table.
    rewards : list[float]
        Accumulated reward per episode for plotting.
    """

    # Create environment with requested configuration
    env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)

    state_space = env.observation_space.n
    action_space = env.action_space.n
    q_table = np.zeros((state_space, action_space), dtype=np.float32)

    episode_rewards: list[float] = []

    for ep in range(episodes):
        state, _ = env.reset(seed=None)
        done = False
        total_reward = 0.0

        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update rule
            best_next = float(np.max(q_table[next_state]))
            td_target = reward + gamma * best_next
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            total_reward += reward
            state = next_state

        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        # Log progress every 1000 episodes
        if (ep + 1) % 1000 == 0:
            avg_last = np.mean(episode_rewards[-1000:])
            print(
                f"Episode {ep + 1:>5d} | Average reward (last 1k): {avg_last:.3f} | ε = {epsilon:.3f}"
            )

    env.close()
    return q_table, episode_rewards


# ------------------------- Visualise & Play ------------------------- #

def plot_rewards(rewards: list[float], save_path: Path | None = None) -> None:
    """Plot the running average rewards."""

    window = 100
    if len(rewards) < window:
        window = len(rewards)

    running_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(8, 4))
    plt.plot(running_avg)
    plt.title(f"Running average reward (window = {window})")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def play(
    q_table: np.ndarray,
    env_name: str = "FrozenLake-v1",
    map_name: str = "4x4",
    is_slippery: bool = True,
) -> float:
    """Let the trained agent play one episode and render it visually (console)."""

    env = gym.make(
        env_name, map_name=map_name, is_slippery=is_slippery, render_mode="human"
    )
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.5)  # Slow down for human observation

    env.close()
    print(f"Agent finished episode with reward: {total_reward}\n")
    return total_reward


# ------------------------------ CLI ------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="Q-learning demo (FrozenLake)")
    parser.add_argument("--episodes", type=int, default=10_000, help="Number of training episodes")
    parser.add_argument("--map", dest="map_name", choices=["4x4", "8x8"], default="4x4", help="FrozenLake map size")
    parser.add_argument("--deterministic", action="store_true", help="Disable slipperiness (is_slippery=False)")
    parser.add_argument("--alpha", type=float, default=0.8, help="Learning rate (α)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor (γ)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate (ε)")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Exploration decay per step")
    parser.add_argument("--min-epsilon", type=float, default=0.01, help="Minimum ε value")
    parser.add_argument("--save-table", default="q_table.npy", help="File to store learned Q-table")
    parser.add_argument("--no-plot", action="store_true", help="Skip reward plot at the end of training")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes used to evaluate policy performance")
    return parser.parse_args()


def main():
    args = parse_args()

    is_slippery = not args.deterministic

    # Evaluate random baseline
    baseline = evaluate_policy(
        None,
        map_name=args.map_name,
        is_slippery=is_slippery,
        episodes=args.eval_episodes,
    )
    print(f"Random policy success rate: {baseline*100:.1f}%  (over {args.eval_episodes} episodes)")

    # Train agent
    q_table, rewards = q_learning(
        map_name=args.map_name,
        is_slippery=is_slippery,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
    )

    learned = evaluate_policy(
        q_table,
        map_name=args.map_name,
        is_slippery=is_slippery,
        episodes=args.eval_episodes,
    )
    print(
        f"\nLearned policy success rate:  {learned*100:.1f}%  (over {args.eval_episodes} episodes)"
    )

    # Save Q-table
    np.save(args.save_table, q_table)
    print(f"Saved Q-table to {args.save_table}")

    if not args.no_plot:
        plot_path = Path(__file__).with_suffix(".png")
        plot_rewards(rewards, save_path=plot_path)

    # Let the agent play once
    _ = play(q_table, map_name=args.map_name, is_slippery=is_slippery)


# --------------------------- Evaluation --------------------------- #

def evaluate_policy(
    q_table: np.ndarray | None,
    env_name: str = "FrozenLake-v1",
    map_name: str = "4x4",
    is_slippery: bool = True,
    episodes: int = 100,
) -> float:
    """Estimate average reward of a policy over multiple episodes.

    If *q_table* is None a random-action baseline is evaluated.
    Returns the mean reward (success rate for FrozenLake).
    """

    env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)
    total_reward = 0.0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            if q_table is None:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                total_reward += reward

    env.close()
    return total_reward / episodes


if __name__ == "__main__":
    main() 