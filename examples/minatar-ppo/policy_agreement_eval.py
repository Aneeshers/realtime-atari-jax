#!/usr/bin/env python3
"""
Evaluate where "worse" PPO checkpoints can be safely used by comparing
their greedy actions to the "best" checkpoint's greedy action on the
exact same states visited by the best policy.

- Rolls out episodes with the best checkpoint's greedy action.
- For each visited state, records which other checkpoints would have
  taken the same action.
- Summarizes "how easy" a state was (e.g., all checkpoints agree) and
  at which earliest checkpoint the action already matches the best.
- Saves a CSV and a couple of plots.

Assumptions:
- Your checkpoints were saved with `pickle.dump(nnx.state(model, nnx.Param), f)`
  as in your training code.
- Environment is a `pgx` MinAtar env (e.g., "minatar-space_invaders").
- ActorCritic architecture matches what you trained.
"""

import argparse
import os
import re
import pickle
from typing import List, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from flax import nnx
import optax

# Optional: if your code lives outside PYTHONPATH, add it (adjust as needed)
import sys
new_path = "/home/ubuntu/tensorflow_test/control/real-timeRL/realtime-atari-jax"
if new_path not in sys.path:
    sys.path.insert(0, new_path)

import pgx  # type: ignore


# -----------------------------
# Model (must match training)
# -----------------------------
def pool_out_dim(n: int, window: int = 2, stride: int = 2, padding: str = "VALID") -> int:
    if padding.upper() == "VALID":
        return (n - window) // stride + 1
    return int(np.ceil(n / stride))


class ActorCritic(nnx.Module):
    def __init__(self, num_actions: int, obs_shape, activation: str = "tanh", *, rngs: nnx.Rngs):
        assert activation in ["relu", "tanh"]
        self.num_actions = num_actions
        self.activation = activation

        H, W, C = obs_shape  # NHWC
        self.conv = nnx.Conv(in_features=C, out_features=32, kernel_size=(2, 2), rngs=rngs)
        self.avg_pool = lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")

        H2 = pool_out_dim(H, 2, 2, "VALID")
        W2 = pool_out_dim(W, 2, 2, "VALID")
        flatten_dim = H2 * W2 * 32

        self.fc = nnx.Linear(flatten_dim, 64, rngs=rngs)

        self.actor_h1 = nnx.Linear(64, 64, rngs=rngs)
        self.actor_h2 = nnx.Linear(64, 64, rngs=rngs)
        self.actor_out = nnx.Linear(64, num_actions, rngs=rngs)

        self.critic_h1 = nnx.Linear(64, 64, rngs=rngs)
        self.critic_h2 = nnx.Linear(64, 64, rngs=rngs)
        self.critic_out = nnx.Linear(64, 1, rngs=rngs)

    def _act(self, x):
        return nnx.relu(x) if self.activation == "relu" else nnx.tanh(x)

    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = self.conv(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)
        x = x.reshape((x.shape[0], -1))
        x = nnx.relu(self.fc(x))

        a = self._act(self.actor_h1(x))
        a = self._act(self.actor_h2(a))
        logits = self.actor_out(a)

        v = self._act(self.critic_h1(x))
        v = self._act(self.critic_h2(v))
        value = self.critic_out(v)

        return logits, jnp.squeeze(value, axis=-1)


# -----------------------------
# Utils
# -----------------------------
def parse_steps_from_path(path: str) -> int:
    m = re.search(r"steps=(\d+)", path)
    return int(m.group(1)) if m else -1


def build_model(env_name: str, activation: str = "tanh", seed: int = 0) -> Tuple[ActorCritic, Any, int]:
    env = pgx.make(env_name)
    num_actions = env.num_actions
    obs_shape = env.observation_shape
    rng = jax.random.PRNGKey(seed)
    rng, sub = jax.random.split(rng)
    model = ActorCritic(num_actions, obs_shape=obs_shape, activation=activation, rngs=nnx.Rngs(sub))
    return model, env, num_actions


def load_params_into_model(model: nnx.Module, ckpt_path: str) -> None:
    """Load nnx.Param state into `model` in-place.

    Works with typical NNX APIs; tries a couple of variants for compatibility.
    """
    with open(ckpt_path, "rb") as f:
        params_state = pickle.load(f)

    # Try common NNX APIs in order
    tried = []
    for apiname in ("set", "update", "merge"):
        fn = getattr(nnx, apiname, None)
        if fn is None:
            continue
        try:
            fn(model, params_state)  # type: ignore[arg-type]
            return
        except Exception as e:
            tried.append((apiname, str(e)))

    # Last resort: try setting using the current state's structure
    current = nnx.state(model, nnx.Param)
    try:
        # Align structure by replacing leaves from `current` with `params_state`
        def replace(_dst, src):
            return src
        replaced = jax.tree_util.tree_map(replace, current, params_state)
        # Try again with set / update, preferring 'set'
        if hasattr(nnx, "set"):
            nnx.set(model, replaced)  # type: ignore[arg-type]
        elif hasattr(nnx, "update"):
            nnx.update(model, replaced)  # type: ignore[arg-type]
        else:
            raise RuntimeError("Could not find nnx.set/nnx.update to assign state.")
        return
    except Exception as e:
        msg = " | ".join([f"{a}: {err}" for a, err in tried]) or "no API worked"
        raise RuntimeError(f"Failed to load params from {ckpt_path}. Tried -> {msg}. Last error: {e}")


def greedy_actions_for_batch(model: nnx.Module, obs: jnp.ndarray) -> np.ndarray:
    logits, _ = model(obs)  # [B, A]
    return np.asarray(jnp.argmax(logits, axis=-1))  # [B]


# -----------------------------
# Core evaluation
# -----------------------------
def evaluate_agreement(
    env_name: str,
    ckpt_paths: List[str],
    num_envs: int = 2,
    seed: int = 0,
    max_episode_len: int = 2000,
    out_dir: str = "./agreement_eval",
):
    os.makedirs(out_dir, exist_ok=True)

    # Sort checkpoints by steps (ascending = "worse" -> "best")
    ckpt_paths = sorted(ckpt_paths, key=parse_steps_from_path)
    steps_list = [parse_steps_from_path(p) for p in ckpt_paths]
    assert steps_list[-1] == max(steps_list), "Last checkpoint should be the best (largest steps)."

    # Build one model per checkpoint (same structure, different params)
    base_model, env, _ = build_model(env_name, seed=seed)
    models = []
    for path in ckpt_paths:
        m, _, _ = build_model(env_name, seed=seed)  # fresh init
        load_params_into_model(m, path)
        models.append(m)

    best_model = models[-1]
    print("models loaded")
    # Initialize env batch
    rng = jax.random.PRNGKey(seed)
    rng, sub = jax.random.split(rng)
    init_keys = jax.random.split(sub, num_envs)
    batched_init = jax.vmap(env.init)
    state = batched_init(init_keys)
    terminated = np.zeros((num_envs,), dtype=bool)
    print("env initialized")
    step_fn = jax.vmap(env.step)
    print("step_fn initialized")
    # Storage
    records: List[Dict[str, Any]] = []

    # Roll out until *all* envs terminate or max_episode_len reached
    t = 0
    while (not bool(np.all(terminated))) and (t < max_episode_len):
        obs = state.observation  # [B, H, W, C]
        # Best policy actions (oracle)
        best_actions = greedy_actions_for_batch(best_model, obs)  # [B]

        # Other checkpoints' actions
        all_actions = []
        for m in models:
            all_actions.append(greedy_actions_for_batch(m, obs))  # list of [B]
        # shape -> [num_ckpts, B]
        all_actions = np.stack(all_actions, axis=0)

        # Record per-env rows
        for env_i in range(num_envs):
            row: Dict[str, Any] = {
                "t": t,
                "env": env_i,
                "oracle_action": int(best_actions[env_i]),
            }
            # Add one column per checkpoint
            for idx, steps in enumerate(steps_list):
                row[f"action_steps={steps}"] = int(all_actions[idx, env_i])
                row[f"match_steps={steps}"] = bool(all_actions[idx, env_i] == best_actions[env_i])

            # How early could we have used a worse policy?
            earliest_ok_idx = None
            for idx in range(len(steps_list)):  # includes best
                if all_actions[idx, env_i] == best_actions[env_i]:
                    earliest_ok_idx = idx
                    break
            row["earliest_ok_ckpt_steps"] = int(steps_list[earliest_ok_idx]) if earliest_ok_idx is not None else -1

            # "easy" if everyone agrees (including best, trivially)
            row["easy_all_agree"] = bool(np.all(all_actions[:, env_i] == best_actions[env_i]))

            records.append(row)

        # Step envs with oracle action
        rng, sub = jax.random.split(rng)
        step_keys = jax.random.split(sub, num_envs)
        state = step_fn(state, jnp.asarray(best_actions), step_keys)
        terminated = np.asarray(state.terminated)
        t += 1
    
    print("records appended")
    # To DataFrame
    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(out_dir, "per_state_agreement.csv")
    df.to_csv(csv_path, index=False)
    print("csv saved")
    # ---- Aggregates ----
    # Category: "needs last k" == earliest_ok is among the last k checkpoints
    n = len(steps_list)

    def needs_last_k(k: int) -> int:
        thresholds = set(steps_list[-k:])
        return int((df["earliest_ok_ckpt_steps"].isin(thresholds)).sum())

    summary = {
        "total_states": int(len(df)),
        "all_versions_ok_states": int(df["easy_all_agree"].sum()),
        "needs_last_1_states": needs_last_k(1),
        "needs_last_2_states": needs_last_k(2),
        "needs_last_3_states": needs_last_k(3) if n >= 3 else None,
        "ckpt_steps_order": steps_list,
        "states_ok_from_ckpt_steps_counts": {
            int(s): int((df["earliest_ok_ckpt_steps"] == s).sum()) for s in steps_list
        },
    }
    print("summary created")
    pd.DataFrame(
        {
            "ckpt_steps": list(summary["states_ok_from_ckpt_steps_counts"].keys()),
            "count": list(summary["states_ok_from_ckpt_steps_counts"].values()),
        }
    ).to_csv(os.path.join(out_dir, "states_ok_from_ckpt_counts.csv"), index=False)
    print("states_ok_from_ckpt_counts.csv saved")
    # Save summary JSON
    import json
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("summary.json saved")
    # ---- Plots ----
    # 1) Bar chart: how many states are okay from each checkpoint
    counts = np.array([summary["states_ok_from_ckpt_steps_counts"][int(s)] for s in steps_list])
    plt.figure(figsize=(8, 4))
    plt.bar([str(s) for s in steps_list], counts)
    plt.xlabel("Earliest checkpoint (steps) that already matches best action")
    plt.ylabel("# states")
    plt.title("How early can we safely use a worse policy?")
    bar_path = os.path.join(out_dir, "earliest_ok_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160)
    plt.close()

    # 2) Line: cumulative share of "easy" states over time (by step index)
    #    Here "time" = rollout step t (aggregated over envs).
    easy_by_t = df.groupby("t")["easy_all_agree"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(easy_by_t["t"].values, easy_by_t["easy_all_agree"].values)
    plt.xlabel("Rollout step t")
    plt.ylabel("Fraction of states where ALL checkpoints agree")
    plt.title("Easiness over rollout")
    line_path = os.path.join(out_dir, "easy_fraction_over_time.png")
    plt.tight_layout()
    plt.savefig(line_path, dpi=160)
    plt.close()
    print("plots saved")
    return {
        "csv_path": csv_path,
        "bar_plot": bar_path,
        "line_plot": line_path,
        "summary_path": os.path.join(out_dir, "summary.json"),
        "ckpt_steps": steps_list,
    }
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="minatar-space_invaders")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--max_episode_len", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default="./agreement_eval")
    parser.add_argument(
        "--ckpts",
        type=str,
        nargs="+",
        required=True,
        help="List of checkpoint paths: oldest -> newest (if unsorted, will be sorted by steps=)",
    )
    args = parser.parse_args()

    results = evaluate_agreement(
        env_name=args.env_name,
        ckpt_paths=args.ckpts,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_len=args.max_episode_len,
        out_dir=args.out_dir,
    )

    print("Saved outputs:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
