#!/usr/bin/env python3
"""Analyze action agreement across PPO checkpoints for MinAtar environments.

This script rolls out the environment using the strongest (latest) checkpoint
to generate a sequence of states, then replays those states through a set of
older checkpoints to study how often their greedy actions match the reference.

Outputs include:
  * A CSV summarising per-state agreement data
  * A JSON file with aggregate statistics
  * A bar chart visualising how many top-N models agree with the reference
"""

import os
import sys
import json
import pickle
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from flax import nnx
from omegaconf import OmegaConf
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Environment path setup (mirrors train_nnx_space.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/ubuntu/tensorflow_test/control/real-timeRL/realtime-atari-jax")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"

import pgx  # noqa: E402  # pylint: disable=wrong-import-position


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pylint: disable=wrong-import-position


# ---------------------------------------------------------------------------
# Shared model components (copied from train_nnx_space.py)
# ---------------------------------------------------------------------------
class Categorical:
    def __init__(self, logits):
        self.logits = logits

    def sample(self, seed):
        return jax.random.categorical(seed, self.logits)

    def log_prob(self, value):
        log_probs = jax.nn.log_softmax(self.logits)
        return jnp.take_along_axis(log_probs, value[..., None], axis=-1).squeeze(-1)

    def entropy(self):
        log_probs = jax.nn.log_softmax(self.logits)
        probs = jax.nn.softmax(self.logits)
        return -(probs * log_probs).sum(axis=-1)


def pool_out_dim(n: int, window: int = 2, stride: int = 2, padding: str = "VALID") -> int:
    if padding.upper() == "VALID":
        return (n - window) // stride + 1
    return int(np.ceil(n / stride))


class ActorCritic(nnx.Module):
    def __init__(self, num_actions: int, obs_shape, activation: str = "tanh", *, rngs: nnx.Rngs):
        assert activation in ["relu", "tanh"]
        self.num_actions = num_actions
        self.activation = activation

        H, W, C = obs_shape
        self.conv = nnx.Conv(in_features=C, out_features=32, kernel_size=(2, 2), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2), padding="VALID")

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class AgreementConfig(BaseModel):
    env_name: str = "minatar-space_invaders"
    seed: int = 0
    num_eval_envs: int = 2
    max_eval_steps: int = 2048
    deterministic: bool = True
    models_dir: str = str(PROJECT_ROOT / "examples" / "minatar-ppo" / "space_models")
    model_paths: List[str] = []
    output_dir: str = str(PROJECT_ROOT / "examples" / "minatar-ppo" / "agreement_analysis")
    csv_name: str = "state_agreement.csv"
    summary_name: str = "summary.json"
    figure_name: str = "agreement_hist.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip("_")).strip("_") or "model"


def infer_step(path: Path) -> int:
    match = re.search(r"steps=(\d+)", path.name)
    return int(match.group(1)) if match else -1


def build_model(env, rng):
    obs_shape = env.observation_shape
    return ActorCritic(env.num_actions, obs_shape=obs_shape, activation="tanh", rngs=nnx.Rngs(rng))


def load_model(env, ckpt_path: Path, rng):
    model = build_model(env, rng)
    with ckpt_path.open("rb") as f:
        param_state = pickle.load(f)
    nnx.update(model, param_state)
    return model


@dataclass
class ModelBundle:
    name: str
    path: Path
    step: int
    model: nnx.Module


def resolve_models(cfg: AgreementConfig, env) -> List[ModelBundle]:
    paths: List[Path]
    if cfg.model_paths:
        paths = [Path(p).expanduser().resolve() for p in cfg.model_paths]
    else:
        search_dir = Path(cfg.models_dir).expanduser().resolve()
        paths = sorted(search_dir.glob("*.ckpt"))
    if len(paths) < 2:
        raise ValueError("At least two checkpoints are required for agreement analysis.")

    bundles: List[ModelBundle] = []
    rng = jax.random.PRNGKey(cfg.seed)
    for path in paths:
        rng, load_rng = jax.random.split(rng)
        model = load_model(env, path, load_rng)
        bundles.append(
            ModelBundle(
                name=sanitize_name(path.stem),
                path=path,
                step=infer_step(path),
                model=model,
            )
        )

    bundles.sort(key=lambda b: (b.step, b.name))
    return bundles


def collect_reference_rollout(cfg: AgreementConfig, env, model_bundle: ModelBundle):
    rng = jax.random.PRNGKey(cfg.seed + 1)
    rng, init_rng = jax.random.split(rng)
    subkeys = jax.random.split(init_rng, cfg.num_eval_envs)
    state = jax.vmap(env.init)(subkeys)

    records = []
    observations = []

    for step in range(cfg.max_eval_steps):
        print(f"step {step} of {cfg.max_eval_steps}")
        terminated = np.array(state.terminated)
        if terminated.all():
            break

        obs = np.array(state.observation)
        logits, _ = model_bundle.model(state.observation)
        if cfg.deterministic:
            actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        else:
            rng, act_rng = jax.random.split(rng)
            actions = Categorical(logits=logits).sample(act_rng).astype(jnp.int32)

        rng, step_rng = jax.random.split(rng)
        keys = jax.random.split(step_rng, cfg.num_eval_envs)
        next_state = jax.vmap(env.step)(state, actions, keys)

        rewards = np.array(next_state.rewards)
        done = np.array(next_state.terminated)
        best_actions = np.array(actions)

        for env_idx in range(cfg.num_eval_envs):
            if terminated[env_idx]:
                continue
            record_index = len(records)
            records.append(
                {
                    "state_index": record_index,
                    "episode": env_idx,
                    "step": step,
                    "best_action": int(best_actions[env_idx]),
                    "reward": float(rewards[env_idx]),
                    "done": bool(done[env_idx]),
                }
            )
            observations.append(obs[env_idx])

        state = next_state

    if not records:
        raise RuntimeError("No states collected. Check max_eval_steps or environment configuration.")

    return records, np.array(observations)


def batched_actions(model: nnx.Module, observations: np.ndarray, batch_size: int = 512) -> np.ndarray:
    actions = []
    total = observations.shape[0]
    for start in range(0, total, batch_size):
        batch = observations[start : start + batch_size]
        logits, _ = model(jnp.asarray(batch))
        batch_actions = jnp.argmax(logits, axis=-1)
        actions.append(np.array(batch_actions))
    return np.concatenate(actions, axis=0)


def compute_agreement(model_bundles: Sequence[ModelBundle], actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    names = [bundle.name for bundle in model_bundles]
    best_name = names[-1]
    best_actions = actions[best_name]
    matrix = np.stack([actions[name] == best_actions for name in names], axis=0)

    match_counts = matrix.sum(axis=0)

    suffix_counts = np.zeros(best_actions.shape[0], dtype=np.int32)
    for idx in range(best_actions.shape[0]):
        count = 0
        for row in reversed(range(matrix.shape[0])):
            if matrix[row, idx]:
                count += 1
            else:
                break
        suffix_counts[idx] = count

    return {
        "match_counts": match_counts,
        "suffix_counts": suffix_counts,
        "match_matrix": matrix,
        "best_actions": best_actions,
        "names": np.array(names),
    }


def summarise_agreement(agreement: Dict[str, np.ndarray]) -> Dict[str, object]:
    names = agreement["names"].tolist()
    total_states = int(agreement["best_actions"].shape[0])
    unique, counts = np.unique(agreement["suffix_counts"], return_counts=True)
    suffix_summary = {int(k): int(v) for k, v in zip(unique, counts)}
    match_counts = agreement["match_counts"]
    mean_matching = float(match_counts.mean())
    frac_all = float(suffix_summary.get(len(names), 0) / total_states)

    return {
        "model_order": names,
        "total_states": total_states,
        "suffix_summary": suffix_summary,
        "average_matching_models": mean_matching,
        "fraction_all_models_agree": frac_all,
    }


def render_histogram(output_path: Path, agreement: Dict[str, np.ndarray]):
    num_models = len(agreement["names"])
    suffix_counts = agreement["suffix_counts"]
    unique, counts = np.unique(suffix_counts, return_counts=True)
    labels = []
    values = []
    for suffix, count in sorted(zip(unique, counts)):
        if suffix == num_models:
            label = "all models"
        elif suffix == 1:
            label = "best only"
        else:
            label = f"top {suffix}"
        labels.append(label)
        values.append(count)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color="#386cb0")
    ax.set_xlabel("Smallest suffix of top checkpoints that agree with latest action")
    ax.set_ylabel("Number of states")
    ax.set_title("Checkpoint agreement across states")
    ax.bar_label(bars, labels=[f"{v} ({v / suffix_counts.size:.1%})" for v in values])
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_dataframe(records: List[Dict[str, object]], actions: Dict[str, np.ndarray], agreement: Dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for name, values in actions.items():
        df[f"action__{name}"] = values
    df["matching_models"] = agreement["match_counts"]
    df["matching_fraction"] = agreement["match_counts"] / len(agreement["names"])
    df["top_suffix_agree"] = agreement["suffix_counts"]
    df["category"] = df["top_suffix_agree"].map(
        lambda k: "all_models" if k == len(agreement["names"]) else ("best_only" if k == 1 else f"top_{k}")
    )
    return df


def main():
    cfg = AgreementConfig(**OmegaConf.to_object(OmegaConf.from_cli()))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = pgx.make(str(cfg.env_name))
    print("env initialized")

    model_bundles = resolve_models(cfg, env)
    actions: Dict[str, np.ndarray] = {}
    print("model bundles resolved")
    reference_bundle = model_bundles[-1]
    print("reference bundle selected")
    print("collecting reference rollout")
    records, observations = collect_reference_rollout(cfg, env, reference_bundle)
    print("reference rollout collected")
    for bundle in model_bundles:
        print(f"batching actions for {bundle.name}")
        actions[bundle.name] = batched_actions(bundle.model, observations)

    agreement = compute_agreement(model_bundles, actions)
    summary = summarise_agreement(agreement)

    df = build_dataframe(records, actions, agreement)

    csv_path = output_dir / cfg.csv_name
    df.to_csv(csv_path, index=False)

    summary_path = output_dir / cfg.summary_name
    summary_path.write_text(json.dumps(summary, indent=2))

    figure_path = output_dir / cfg.figure_name
    render_histogram(figure_path, agreement)

    print("Agreement analysis complete.")
    print(f"- States analysed: {summary['total_states']}")
    print(f"- Fraction all models agree: {summary['fraction_all_models_agree']:.2%}")
    print(f"- Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()


