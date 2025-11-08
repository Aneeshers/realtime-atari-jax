"""This PPO implementation is modified from PureJaxRL:

  https://github.com/luchris429/purejaxrl

Please refer to their work if you use this example in your research."""

import sys
import time
import math
import shutil
import pickle
from functools import partial
from typing import NamedTuple, Literal

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb


import sys
import os

# Add the path to sys.path for the current Python session
new_path = "/home/ubuntu/tensorflow_test/control/real-timeRL/realtime-atari-jax"

# Add to sys.path if not already there
if new_path not in sys.path:
    sys.path.insert(0, new_path)

# Also set PYTHONPATH for any subprocesses
os.environ["PYTHONPATH"] = f"{new_path}:{os.environ.get('PYTHONPATH', '')}"

# Verify it worked
print("Python path updated:")
print(f"sys.path includes: {new_path}")
print(f"PYTHONPATH env var: {os.environ['PYTHONPATH']}")

import pgx
from pgx.experimental import auto_reset


# -----------------------------
# Simple Categorical distribution wrapper using JAX built-ins
# -----------------------------
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


# -----------------------------
# Config
# -----------------------------
class PPOConfig(BaseModel):
    env_name: Literal[
        "minatar-breakout",
        "minatar-freeway",
        "minatar-space_invaders",
        "minatar-asterix",
        "minatar-seaquest",
    ] = "minatar-space_invaders"
    seed: int = 0
    lr: float = 0.0003
    num_envs: int = 4096
    num_eval_envs: int = 100
    num_steps: int = 128
    total_timesteps: int = 20_000_000
    update_epochs: int = 3
    minibatch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    wandb_project: str = "pgx-minatar-ppo"
    save_model: bool = True
    out_models_dir: str = "/home/ubuntu/tensorflow_test/control/real-timeRL/realtime-atari-jax/examples/minatar-ppo/space_models"

    class Config:
        extra = "forbid"


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args, file=sys.stderr)

env = pgx.make(str(args.env_name))
num_updates = args.total_timesteps // args.num_envs // args.num_steps
num_minibatches = args.num_envs * args.num_steps // args.minibatch_size


# -----------------------------
# NNX Actor-Critic
# -----------------------------
def pool_out_dim(n: int, window: int = 2, stride: int = 2, padding: str = "VALID") -> int:
    # Matches flax.linen/nnx pooling semantics for VALID padding
    if padding.upper() == "VALID":
        return (n - window) // stride + 1
    # Fallback (not used here)
    return math.ceil(n / stride)


class ActorCritic(nnx.Module):
    def __init__(self, num_actions: int, obs_shape, activation: str = "tanh", *, rngs: nnx.Rngs):
        assert activation in ["relu", "tanh"]
        self.num_actions = num_actions
        self.activation = activation

        H, W, C = obs_shape  # NHWC expected by flax.nnx.Conv
        # Convolution (channels-last). Default padding is 'SAME'.
        self.conv = nnx.Conv(in_features=C, out_features=32, kernel_size=(2, 2), rngs=rngs)

        # AvgPool params are fixed; keep a partial for clean callsites
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2), padding="VALID")

        # After conv ('SAME') + avg_pool('VALID', 2x2, stride 2) the spatial dims become:
        H2 = pool_out_dim(H, 2, 2, "VALID")
        W2 = pool_out_dim(W, 2, 2, "VALID")
        flatten_dim = H2 * W2 * 32

        # Shared torso
        self.fc = nnx.Linear(flatten_dim, 64, rngs=rngs)

        # Actor head: 64 -> 64 -> 64 -> num_actions (two hidden layers like original)
        self.actor_h1 = nnx.Linear(64, 64, rngs=rngs)
        self.actor_h2 = nnx.Linear(64, 64, rngs=rngs)
        self.actor_out = nnx.Linear(64, num_actions, rngs=rngs)

        # Critic head: 64 -> 64 -> 64 -> 1 (two hidden layers like original)
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
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nnx.relu(self.fc(x))

        a = self._act(self.actor_h1(x))
        a = self._act(self.actor_h2(a))
        logits = self.actor_out(a)

        v = self._act(self.critic_h1(x))
        v = self._act(self.critic_h2(v))
        value = self.critic_out(v)

        return logits, jnp.squeeze(value, axis=-1)


# -----------------------------
# Optimizer (Optax via NNX wrapper)
# -----------------------------
tx = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm),
    optax.adam(args.lr, eps=1e-5),
)


# -----------------------------
# Rollout container
# -----------------------------
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


def save_checkpoint(model: nnx.Module, step: int) -> str:
    checkpoint_path = os.path.join(
        args.out_models_dir,
        f"{args.env_name}-seed={args.seed}-steps={step}.ckpt",
    )
    with open(checkpoint_path, "wb") as f:
        pickle.dump(nnx.state(model, nnx.Param), f)
    return checkpoint_path


# -----------------------------
# Update step (collect + optimize), jitted with NNX
# -----------------------------
def make_update_step():
    step_fn = jax.vmap(auto_reset(env.step, env.init))

    @nnx.jit(donate_argnames=("model", "optimizer"))
    def _update_step(model: nnx.Module,
                     optimizer: nnx.Optimizer,
                     env_state,
                     last_obs,
                     rng):
        # -------- Collect trajectories --------
        def _env_step(runner_state, _):
            model, optimizer, env_state, last_obs, rng = runner_state

            # Policy
            rng, _rng = jax.random.split(rng)
            logits, value = model(last_obs)
            pi = Categorical(logits=logits)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # Env step
            rng, _rng = jax.random.split(rng)
            keys = jax.random.split(_rng, env_state.observation.shape[0])
            env_state = step_fn(env_state, action, keys)

            transition = Transition(
                env_state.terminated,
                action,
                value,
                jnp.squeeze(env_state.rewards),
                log_prob,
                last_obs,
            )
            runner_state = (model, optimizer, env_state, env_state.observation, rng)
            return runner_state, transition

        runner_state = (model, optimizer, env_state, last_obs, rng)
        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, length=args.num_steps)

        # -------- Advantage / targets (GAE) --------
        model, optimizer, env_state, last_obs, rng = runner_state
        _, last_val = model(last_obs)

        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + args.gamma * next_value * (1 - done) - value
            gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
            return (gae, value), gae

        (_, _), advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        targets = advantages + traj_batch.value

        # -------- SGD epochs --------
        def _update_epoch(update_state, _):
            model, optimizer, traj_batch, advantages, targets, rng = update_state

            def _update_minibatch(state, minibatch):
                model, optimizer = state
                mb_traj, mb_adv, mb_targets = minibatch

                def _loss_fn(model: nnx.Module, traj: Transition, gae, targets):
                    # Re-run policy
                    logits, value = model(traj.obs)
                    pi = Categorical(logits=logits)
                    log_prob = pi.log_prob(traj.action)

                    # Value loss (clipped)
                    value_pred_clipped = traj.value + (value - traj.value).clip(-args.clip_eps, args.clip_eps)
                    v_loss_unclipped = jnp.square(value - targets)
                    v_loss_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

                    # Policy loss (clipped)
                    ratio = jnp.exp(log_prob - traj.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                    # Entropy bonus
                    entropy = pi.entropy().mean()

                    total = loss_actor + args.vf_coef * value_loss - args.ent_coef * entropy
                    return total, (value_loss, loss_actor, entropy)

                # Compute grads w.r.t. model Params
                (total_loss, aux), grads = nnx.value_and_grad(
                    _loss_fn, has_aux=True, argnums=nnx.DiffState(0, nnx.Param)
                )(model, mb_traj, mb_adv, mb_targets)

                # Optax step via NNX Optimizer (updates model in-place)
                optimizer.update(model, grads)

                return (model, optimizer), (total_loss, aux)

            # Shuffle + minibatch
            rng, _rng = jax.random.split(rng)
            batch_size = args.minibatch_size * num_minibatches
            assert batch_size == args.num_steps * args.num_envs, "batch size must equal steps * envs"

            batch = (traj_batch, advantages, targets)
            batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            permutation = jax.random.permutation(_rng, batch_size)
            shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                shuffled,
            )

            (model, optimizer), losses = jax.lax.scan(_update_minibatch, (model, optimizer), minibatches)
            update_state = (model, optimizer, traj_batch, advantages, targets, rng)
            return update_state, losses

        update_state = (model, optimizer, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, length=args.update_epochs)

        model, optimizer, _, _, _, rng = update_state
        runner_state = (model, optimizer, env_state, last_obs, rng)
        return runner_state, loss_info

    return _update_step


# -----------------------------
# Evaluation (greedy sample)
# -----------------------------
@nnx.jit
def evaluate(model: nnx.Module, rng_key):
    step_fn = jax.vmap(env.step)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, args.num_eval_envs)
    state = jax.vmap(env.init)(subkeys)
    R = jnp.zeros_like(state.rewards)

    def cond_fn(tup):
        state, _, _ = tup
        return ~state.terminated.all()

    def loop_fn(tup):
        state, R, rng_key = tup
        logits, _value = model(state.observation)
        pi = Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        keys = jax.random.split(_rng, state.observation.shape[0])
        state = step_fn(state, action, keys)
        return state, R + state.rewards, rng_key

    state, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key))
    return R.mean()


# -----------------------------
# Training Loop
# -----------------------------
def train(rng):
    tt = 0.0
    st = time.time()

    # Model + optimizer
    rng, _rng = jax.random.split(rng)
    obs_shape = env.observation_shape
    model = ActorCritic(env.num_actions, obs_shape=obs_shape, activation="tanh", rngs=nnx.Rngs(_rng))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Update function
    update_step = make_update_step()

    # Init envs
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, args.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(reset_rng)
    last_obs = env_state.observation

    rng, _rng = jax.random.split(rng)
    runner_state = (model, optimizer, env_state, last_obs, _rng)

    # Warmup (compile)
    _, _ = update_step(*runner_state)

    # initial evaluation
    et = time.time()
    tt += et - st
    rng, _rng = jax.random.split(rng)
    eval_R = evaluate(runner_state[0], _rng)
    steps = 0
    training_total = args.num_envs * args.num_steps * num_updates
    checkpoint_targets = []
    checkpoint_paths = {}
    if args.save_model:
        os.makedirs(args.out_models_dir, exist_ok=True)
        base_interval = max(1, math.ceil(training_total / 4))
        checkpoint_targets = [min(training_total, base_interval * i) for i in range(1, 4)]
        checkpoint_targets.append(training_total)
        checkpoint_targets = sorted(set(checkpoint_targets))
    log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
    print(log)
    wandb.log(log)
    st = time.time()

    for _ in range(num_updates):
        runner_state, loss_info = update_step(*runner_state)
        model, optimizer, env_state, last_obs, rng = runner_state
        steps += args.num_envs * args.num_steps

        # evaluation
        et = time.time()
        tt += et - st
        rng, _rng = jax.random.split(rng)
        eval_R = evaluate(model, _rng)
        log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
        print(log)
        wandb.log(log)
        st = time.time()
        if args.save_model:
            for target in checkpoint_targets:
                if steps >= target and target not in checkpoint_paths:
                    checkpoint_paths[target] = save_checkpoint(model, target)

    if args.save_model:
        for target in checkpoint_targets:
            if steps >= target and target not in checkpoint_paths:
                checkpoint_paths[target] = save_checkpoint(runner_state[0], target)

    return runner_state, checkpoint_paths  # (model, optimizer, env_state, last_obs, rng)



if __name__ == "__main__":
    wandb.init(project=args.wandb_project, config=args.dict())
    rng = jax.random.PRNGKey(args.seed)
    out, checkpoint_paths = train(rng)
    if args.save_model:
        model = out[0]
        if checkpoint_paths:
            final_step = max(checkpoint_paths)
            shutil.copyfile(
                checkpoint_paths[final_step],
                os.path.join(
                    args.out_models_dir,
                    f"{args.env_name}-seed={args.seed}.ckpt",
                ),
            )
        else:
            # Save only learnable parameters (nnx.Param state) like Haiku params
            os.makedirs(args.out_models_dir, exist_ok=True)
            with open(
                os.path.join(
                    args.out_models_dir,
                    f"{args.env_name}-seed={args.seed}.ckpt",
                ),
                "wb",
            ) as f:
                pickle.dump(nnx.state(model, nnx.Param), f)


    