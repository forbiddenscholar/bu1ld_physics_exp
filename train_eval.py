import os
import json
import subprocess
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from network import ActorCriticDynamics
from buffer import RolloutBuffer


TRAIN_TIMESTEPS = 25_000
BUFFER_SIZE = 512
BATCH_SIZE = 64
UPDATE_EPOCHS = 10
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VF_COEF = 0.5
AUX_LAMBDA = 0.1
EVAL_EPISODES = 20
TRAIN_SEED = 42
EVAL_SEED_START = 1000


def evaluate(model, env, num_episodes=20, seed_start=1000):
    """
    Evaluates policy deterministically.
    Computes: Mean Return, RMS Theta, Action Effort, and Forward Dynamics MSE.
    """
    returns = []
    all_thetas = []
    all_actions = []
    dyn_errors = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        ep_return = 0.0
        done = False

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Deterministic action: mean of Gaussian policy
            with torch.no_grad():
                features = model.get_features(obs_tensor)
                action_mean = model.actor_mean(features)
                action = action_mean.squeeze(0).numpy()
                
                # Clip to environment action space [-3.0, 3.0]
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                # Forward dynamics prediction: ŝ_{t+1}
                action_tensor = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
                pred_next_state = model.predict_next_state(features, action_tensor, detach_trunk=True)
                pred_next_state = pred_next_state.squeeze(0).numpy()

            # Record step dynamics
            all_thetas.append(obs[1])          # obs[1] is pole angle theta (rad)
            all_actions.append(action[0])       # action force

            # Step simulator
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

            # Dynamics prediction error
            dyn_errors.append(np.mean((pred_next_state - next_obs) ** 2))
            obs = next_obs

        returns.append(ep_return)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "rms_theta": float(np.sqrt(np.mean(np.array(all_thetas) ** 2))),
        "action_effort": float(np.mean(np.array(all_actions) ** 2)),
        "forward_mse": float(np.mean(dyn_errors)),
    }


def train(agent_type="predictive", seed=42):
    """
    Trains an agent for exactly 25,000 interaction steps, as discussed in the protocol.
    agent_type: 'predictive' (aux loss regularizes trunk)
                'baseline' (trunk detached from dynamics loss)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("InvertedPendulum-v4")
    obs, _ = env.reset(seed=seed)

    model = ActorCriticDynamics(state_dim=4, action_dim=1, aux_lambda=AUX_LAMBDA)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    buffer = RolloutBuffer(BUFFER_SIZE, state_dim=4, action_dim=1, gamma=GAMMA, gae_lambda=GAE_LAMBDA)

    total_steps = 0
    num_updates = TRAIN_TIMESTEPS // BUFFER_SIZE

    print(f"\n--- Starting {agent_type.upper()} Training ({TRAIN_TIMESTEPS} steps) ---")

    for update in range(1, num_updates + 1):
        buffer.reset()

        # 1. Rollout Collection
        for _ in range(BUFFER_SIZE):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, _, value, _ = model.get_action_and_value(obs_tensor)
                
            action_np = action.squeeze(0).numpy()
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            total_steps += 1

            # Episode termination handling
            done = terminated or truncated
            buffer.add(
                state=obs,
                action=action_np,
                reward=reward,
                done=done,
                value=value.item(),
                log_prob=log_prob.item(),
                next_state=next_obs
            )

            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

        # 2. Bootstrap Value & Compute GAE
        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            _, _, _, last_value, _ = model.get_action_and_value(last_obs_tensor)
        buffer.compute_gae(last_value.item(), done)

        # 3. PPO Update Epochs
        for _ in range(UPDATE_EPOCHS):
            for b_states, b_actions, b_old_log_probs, b_advantages, b_returns, b_next_states in buffer.get_batches(BATCH_SIZE):
                _, new_log_probs, _, new_values, b_features = model.get_action_and_value(b_states, action=b_actions)

                # Ratio and clipped surrogate policy loss
                log_ratio = new_log_probs - b_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function regression loss
                value_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

                # Auxiliary dynamics MSE loss
                if agent_type == "predictive":
                    # Gradients flow backward into the shared trunk
                    pred_next_states = model.predict_next_state(b_features, b_actions, detach_trunk=False)
                    dyn_loss = torch.mean((pred_next_states - b_next_states) ** 2)
                    total_loss = policy_loss + (VF_COEF * value_loss) + (AUX_LAMBDA * dyn_loss)
                else:
                    # Baseline: detach trunk so auxiliary loss NEVER influences policy representations
                    pred_next_states = model.predict_next_state(b_features, b_actions, detach_trunk=True)
                    dyn_loss = torch.mean((pred_next_states - b_next_states) ** 2)
                    total_loss = policy_loss + (VF_COEF * value_loss) + (1.0 * dyn_loss)

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        if update % 10 == 0 or update == num_updates:
            print(f"Step {total_steps}/{TRAIN_TIMESTEPS} | Policy Loss: {policy_loss.item():.4f} | Dyn Loss: {dyn_loss.item():.6f}")

    env.close()
    return model


def main():
    # Retrieve Git Commit SHA
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        git_sha = "uncommitted_local"

    # Environment Setup
    test_env = gym.make("InvertedPendulum-v4")
    nominal_pole_mass = float(test_env.unwrapped.model.body("pole").mass[0])
    test_env.close()

    results_data = {}

    for agent_type in ["baseline", "predictive"]:
        model = train(agent_type=agent_type, seed=TRAIN_SEED)

        # 1. Nominal Evaluation
        nom_env = gym.make("InvertedPendulum-v4")
        nom_metrics = evaluate(model, nom_env, num_episodes=EVAL_EPISODES, seed_start=EVAL_SEED_START)
        nom_env.close()

        # 2. Shifted Evaluation (2x Pole Mass)
        shift_env = gym.make("InvertedPendulum-v4")
        shift_env.unwrapped.model.body("pole").mass[0] = nominal_pole_mass * 2.0
        shift_metrics = evaluate(model, shift_env, num_episodes=EVAL_EPISODES, seed_start=EVAL_SEED_START)
        shift_env.close()

        # Compute Primary Comparison Metrics
        r_nom = nom_metrics["mean_return"]
        r_shift = shift_metrics["mean_return"]
        prr = (r_shift / r_nom) * 100.0 if r_nom > 0 else 0.0
        delta_r = r_nom - r_shift

        results_data[agent_type] = {
            "nominal_return": r_nom,
            "nominal_std": nom_metrics["std_return"],
            "shifted_return": r_shift,
            "shifted_std": shift_metrics["std_return"],
            "prr_percent": prr,
            "delta_r": delta_r,
            "shifted_rms_theta": shift_metrics["rms_theta"],
            "shifted_action_effort": shift_metrics["action_effort"],
            "shifted_forward_mse": shift_metrics["forward_mse"]
        }

    # Summary Display
    print("\n" + "=" * 80)
    print("                 25k SANITY GATE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<28} | {'Baseline':<22} | {'Predictive (λ=0.1)':<22}")
    print("-" * 80)
    print(f"{'Nominal Return':<28} | {results_data['baseline']['nominal_return']:<22.2f} | {results_data['predictive']['nominal_return']:<22.2f}")
    print(f"{'Shifted Return (2x mass)':<28} | {results_data['baseline']['shifted_return']:<22.2f} | {results_data['predictive']['shifted_return']:<22.2f}")
    print(f"{'PRR (%) [Higher is better]':<28} | {results_data['baseline']['prr_percent']:<22.2f}% | {results_data['predictive']['prr_percent']:<22.2f}%")
    print(f"{'ΔR [Lower is better]':<28} | {results_data['baseline']['delta_r']:<22.2f} | {results_data['predictive']['delta_r']:<22.2f}")
    print(f"{'RMS Theta (rad)':<28} | {results_data['baseline']['shifted_rms_theta']:<22.4f} | {results_data['predictive']['shifted_rms_theta']:<22.4f}")
    print(f"{'Action Effort (u^2)':<28} | {results_data['baseline']['shifted_action_effort']:<22.4f} | {results_data['predictive']['shifted_action_effort']:<22.4f}")
    print(f"{'Forward Model MSE':<28} | {results_data['baseline']['shifted_forward_mse']:<22.6f} | {results_data['predictive']['shifted_forward_mse']:<22.6f}")
    print("=" * 80)

    # Save Output JSON
    output_payload = {
        "metadata": {
            "protocol_version": "v3",
            "git_commit_sha": git_sha,
            "environment": "InvertedPendulum-v4",
            "packages": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "gymnasium": gym.__version__,
            }
        },
        "sanity_gate_config": {
            "train_steps": TRAIN_TIMESTEPS,
            "train_seed": TRAIN_SEED,
            "eval_episodes": EVAL_EPISODES,
            "lambda_aux": AUX_LAMBDA,
            "nominal_pole_mass": nominal_pole_mass,
            "shifted_pole_mass": nominal_pole_mass * 2.0
        },
        "results": results_data
    }

    with open("sanity_gate_results.json", "w") as f:
        json.dump(output_payload, f, indent=2)

    print("\nSaved reproducible artifact to: sanity_gate_results.json")

if __name__ == "__main__":
    main()