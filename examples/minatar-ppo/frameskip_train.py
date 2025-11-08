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

import sys
import math

import jax.numpy as jnp
from typing import Literal, Optional

import jax
from jax import numpy as jnp

import pgx
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from pydantic import BaseModel
import time
import math
import pickle
from functools import partial
from typing import NamedTuple, Literal

import optax
from flax import nnx
import wandb
from pgx.experimental import auto_reset



def get_sizes(state):
    try:
        size = len(state.current_player)
        width = math.ceil(math.sqrt(size - 0.1))
        if size - (width - 1) ** 2 >= width:
            height = width
        else:
            height = width - 1
    except TypeError:
        size = 1
        width = 1
        height = 1
    return size, width, height


def get_cmap(n_channels):
    # import seaborn as sns  # type: ignore
    # return cmap = sns.color_palette("cubehelix", n_channels)
    assert n_channels in (4, 6, 7, 10)
    if n_channels == 4:
        return [(0.08605633600581405, 0.23824692404212, 0.30561236308077167), (0.32927729263408284, 0.4762845556584382, 0.1837155549758328), (0.8146245329198283, 0.49548316572322215, 0.5752525936416857), (0.7587183008012618, 0.7922069335474338, 0.9543861221913403)]
    elif n_channels == 6:
        return [(0.10231025194333628, 0.13952898866828906, 0.2560120319409181), (0.10594361078604106, 0.3809739011595331, 0.27015111282899046), (0.4106130272672762, 0.48044780541672255, 0.1891154277778484), (0.7829183382530567, 0.48158303462490826, 0.48672451968362596), (0.8046168329276406, 0.6365733569301846, 0.8796578402926125), (0.7775608374378459, 0.8840392521212448, 0.9452007992345052)]
    elif n_channels == 7:
        return [(0.10419418740482515, 0.11632019220053316, 0.2327552016195138), (0.08523511613408935, 0.32661779003565533, 0.2973201282529313), (0.26538761550634205, 0.4675654910052002, 0.1908220644759285), (0.6328422475018423, 0.4747981096220677, 0.29070209208025455), (0.8306875710682655, 0.5175161303658079, 0.6628221028832032), (0.7779565181455343, 0.7069421942599752, 0.9314406084043191), (0.7964528047840354, 0.908668973545918, 0.9398253500983916)]
    elif n_channels == 10:
        return [(0.09854228363950114, 0.07115215572295082, 0.16957891809124037), (0.09159726558869188, 0.20394337960213008, 0.29623965888210324), (0.09406611799930162, 0.3578871412608098, 0.2837709711722866), (0.23627685553553793, 0.46114369021199075, 0.19770731888985724), (0.49498740849493095, 0.4799034869159042, 0.21147789468974837), (0.7354526513473981, 0.4748861903571046, 0.40254094042448907), (0.8325928529853291, 0.5253446757844744, 0.6869376931865354), (0.7936920632275369, 0.6641337211433709, 0.9042311843062529), (0.7588424692372241, 0.8253990353420474, 0.9542699331220588), (0.8385645211683802, 0.9411869386771845, 0.9357655639413166)]


# /home/ubuntu/tensorflow_test/control/real-timeRL/realtime-atari-jax/pgx/minatar/utils.py

def visualize_minatar(state, savefile=None, fmt="svg", dpi=160):
    # Modified from https://github.com/kenjyoung/MinAtar
    try:
        import matplotlib.colors as colors  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        sys.stderr.write("MinAtar environment requires matplotlib for visualization. Please install matplotlib.")
        sys.exit(1)

    obs = state.observation
    n_channels = obs.shape[-1]
    cmap = get_cmap(n_channels)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels + 2)]
    norm = colors.BoundaryNorm(bounds, n_channels + 1)
    size, w, h = get_sizes(state)
    fig, ax = plt.subplots(h, w)
    n_channels = obs.shape[-1]
    if size == 1:
        numerical_state = (
            jnp.amax(
                obs * jnp.reshape(jnp.arange(n_channels) + 1, (1, 1, -1)), 2
            )
            + 0.5
        )
        ax.imshow(numerical_state, cmap=cmap, norm=norm, interpolation="none")
        ax.set_axis_off()
    else:
        for j in range(size):
            numerical_state = (
                jnp.amax(
                    obs[j] * jnp.reshape(jnp.arange(n_channels) + 1, (1, 1, -1)),
                    2,
                )
                + 0.5
            )
            if h == 1:
                ax[j].imshow(numerical_state, cmap=cmap, norm=norm, interpolation="none")
                ax[j].set_axis_off()
            else:
                ax[j // w, j % w].imshow(numerical_state, cmap=cmap, norm=norm, interpolation="none")
                ax[j // w, j % w].set_axis_off()

    if savefile is None:
        # Return in-memory image
        if fmt == "svg":
            from io import StringIO
            sio = StringIO()
            plt.savefig(sio, format="svg", bbox_inches="tight")
            plt.close(fig)
            return sio.getvalue()  # str (SVG markup)
        else:
            from io import BytesIO
            bio = BytesIO()
            plt.savefig(bio, format=fmt, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            bio.seek(0)
            return bio.getvalue()  # bytes (e.g., PNG)
    else:
        plt.savefig(savefile, format=fmt, bbox_inches="tight", dpi=(None if fmt == "svg" else dpi))
        plt.close(fig)
        return savefile
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
    ] = "minatar-breakout"
    seed: int = 0
    lr: float = 0.0003
    num_envs: int = 4096
    num_eval_envs: int = 100
    num_steps: int = 128
    total_timesteps: int = 20_000_000
    frame_skip: int = 1
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
    gpu_id: int = 0
    

    class Config:
        extra = "forbid"

from breakout_frame_skip import MinAtarBreakout
from freeway_frame_skip import MinAtarFreeway
# In Jupyter, directly create the config instead of parsing CLI args
# You can override any default values here

"""This PPO implementation is modified from PureJaxRL:

    https://github.com/luchris429/purejaxrl

Please refer to their work if you use this example in your research."""




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
        runner_state, traj_batch = nnx.scan(_env_step, in_axes=(nnx.Carry, None), out_axes=(nnx.Carry, 0), length=ppo_args.num_steps)(runner_state, None)

        # -------- Advantage / targets (GAE) --------
        model, optimizer, env_state, last_obs, rng = runner_state
        _, last_val = model(last_obs)

        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + ppo_args.gamma * next_value * (1 - done) - value
            gae = delta + ppo_args.gamma * ppo_args.gae_lambda * (1 - done) * gae
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
                    value_pred_clipped = traj.value + (value - traj.value).clip(-ppo_args.clip_eps, ppo_args.clip_eps)
                    v_loss_unclipped = jnp.square(value - targets)
                    v_loss_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

                    # Policy loss (clipped)
                    ratio = jnp.exp(log_prob - traj.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - ppo_args.clip_eps, 1.0 + ppo_args.clip_eps) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                    # Entropy bonus
                    entropy = pi.entropy().mean()

                    total = loss_actor + ppo_args.vf_coef * value_loss - ppo_args.ent_coef * entropy
                    return total, (value_loss, loss_actor, entropy)

                # Compute grads w.r.t. model Params
                (total_loss, aux), grads = nnx.value_and_grad(
                    _loss_fn, has_aux=True, argnums=nnx.DiffState(0, nnx.Param)
                )(model, mb_traj, mb_adv, mb_targets)

                # Optax step via NNX Optimizer (updates model in-place)
                optimizer.update(grads=grads)

                return (model, optimizer), (total_loss, aux)

            # Shuffle + minibatch
            rng, _rng = jax.random.split(rng)
            batch_size = ppo_args.minibatch_size * num_minibatches
            assert batch_size == ppo_args.num_steps * ppo_args.num_envs, "batch size must equal steps * envs"

            batch = (traj_batch, advantages, targets)
            batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            permutation = jax.random.permutation(_rng, batch_size)
            shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                shuffled,
            )

            (model, optimizer), losses = nnx.scan(_update_minibatch, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))((model, optimizer), minibatches)
            update_state = (model, optimizer, traj_batch, advantages, targets, rng)
            return update_state, losses

        update_state = (model, optimizer, traj_batch, advantages, targets, rng)
        update_state, loss_info = nnx.scan(_update_epoch, in_axes=(nnx.Carry, None), out_axes=(nnx.Carry, 0), length=ppo_args.update_epochs)(update_state, None)

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
    subkeys = jax.random.split(sub_key, ppo_args.num_eval_envs)
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
    reset_rng = jax.random.split(_rng, ppo_args.num_envs)
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
    log = {"sec": tt, f"{ppo_args.env_name}/eval_R": float(eval_R), "steps": steps}
    print(log)

    # Only log to wandb if initialized
    if wandb.run is not None:
        wandb.log(log)
    st = time.time()

    for _ in range(num_updates):
        runner_state, loss_info = update_step(*runner_state)
        model, optimizer, env_state, last_obs, rng = runner_state
        steps += ppo_args.num_envs * ppo_args.num_steps

        # evaluation
        et = time.time()
        tt += et - st
        rng, _rng = jax.random.split(rng)
        eval_R = evaluate(model, _rng)
        log = {"sec": tt, f"{ppo_args.env_name}/eval_R": float(eval_R), "steps": steps}
        print(log)

        # Only log to wandb if initialized
        if wandb.run is not None:
            wandb.log(log)
        st = time.time()

    return runner_state  # (model, optimizer, env_state, last_obs, rng)


# -----------------------------
# Run training (for notebook execution)
# -----------------------------
# Or run without wandb

from tqdm import tqdm
from pgx.minatar.utils import visualize_minatar  # patched version supports fmt="png"
import io
import imageio.v2 as imageio


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def build_model(env, rng):
    obs_shape = env.observation_shape
    return ActorCritic(env.num_actions, obs_shape=obs_shape, activation="tanh", rngs=nnx.Rngs(rng))

def load_model(env, ckpt_path: str, rng):
    """Recreate module and load nnx.Param state."""
    model = build_model(env, rng)
    with open(ckpt_path, "rb") as f:
        param_state = pickle.load(f)
    nnx.update(model, param_state)
    return model

def eval_rollout_and_save_video(model: nnx.Module,
                                env,
                                rng_key,
                                num_envs_to_render: int = 16,
                                max_steps: int = 500,
                                fps: int = 8,
                                output_gif: str = "eval.gif"):
    """
    Runs a non-jitted rollout for rendering & saves GIF locally.
    """
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, num_envs_to_render)
    state = jax.vmap(env.init)(subkeys)
    total_R = jnp.zeros_like(state.rewards)
    frames_png = []

    # tqdm progress bar
    for t in tqdm(range(max_steps), desc="Evaluating rollout", ncols=80):
        print(f"Step {t}")
        # Render frame
        png_bytes = visualize_minatar(state, savefile=None, fmt="png", dpi=160)
        frames_png.append(png_bytes)

        # Policy step
        logits, _ = model(state.observation)
        rng_key, _rng = jax.random.split(rng_key)
        action = Categorical(logits).sample(seed=_rng)

        # Env step
        rng_key, _rng = jax.random.split(rng_key)
        keys = jax.random.split(_rng, state.observation.shape[0])
        state = jax.vmap(env.step)(state, action, keys)

        total_R = total_R + state.rewards
        if bool(state.terminated.all()):
            break

    # Convert PNGs → GIF
    print(f"\nSaving video to {output_gif} ...")
    imgs = [imageio.imread(io.BytesIO(b)) for b in frames_png]
    imgs = [im[..., :3] if im.ndim == 3 and im.shape[-1] == 4 else im for im in imgs]
    imageio.mimsave(output_gif, imgs, fps=fps)
    print(f"Saved {len(imgs)} frames to {output_gif}")
    print(f"Mean reward: {float(total_R.mean()):.3f}")
    return float(total_R.mean())

# -------------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------------

# set up args:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="minatar-freeway")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--total_timesteps", type=int, default=300000000)
parser.add_argument("--frame_skip", type=int, default=3)
parser.add_argument("--save_model", type=bool, default=True)
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for training (0 or 1)")
args = parser.parse_args()
ppo_args = PPOConfig(**vars(args))
print(f"Config: {ppo_args}")

# Set up GPU selection
print(f"Available JAX devices: {jax.devices()}")
if ppo_args.gpu_id >= len(jax.devices()):
    print(f"Warning: GPU ID {ppo_args.gpu_id} not available. Using device 0 instead.")
    device_id = 0
else:
    device_id = ppo_args.gpu_id

# Set the default device for JAX operations
jax.config.update('jax_default_device', jax.devices()[device_id])
print(f"Using device: {jax.devices()[device_id]}")

env = pgx.make(str(ppo_args.env_name))
if ppo_args.env_name == "minatar-freeway":
    print("using custom env")
    env = MinAtarFreeway(
        use_minimal_action_set=True,
        sticky_action_prob=0.1,
        frame_skip=ppo_args.frame_skip,
    )
if ppo_args.env_name == "minatar-breakout":
    env = MinAtarBreakout(
        use_minimal_action_set=True,
        sticky_action_prob=0.1,
        frame_skip=ppo_args.frame_skip,
    )
tx = optax.chain(
    optax.clip_by_global_norm(ppo_args.max_grad_norm),
    optax.adam(ppo_args.lr, eps=1e-5),
)

num_updates = ppo_args.total_timesteps // ppo_args.num_envs // ppo_args.num_steps
num_minibatches = ppo_args.num_envs * ppo_args.num_steps // ppo_args.minibatch_size
wandb.init(
    project=f"ppo-{ppo_args.env_name}-frameskip",        # "pgx-minatar-ppo" by default
    name=f"{ppo_args.env_name}-frameskip{ppo_args.frame_skip}",
    config=ppo_args.dict() if hasattr(ppo_args, "dict") else vars(ppo_args),
)
# Start training
print("Starting training...")
rng = jax.random.PRNGKey(ppo_args.seed)
runner_state = train(rng)
# save it to root directory
model_dir = f"./minatar-ppo-models/{ppo_args.env_name}/"
    # make model directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# Save model if desired
if ppo_args.save_model:
    model = runner_state[0]
    # Save only learnable parameters
    
    
    with open(f"{model_dir}/{ppo_args.env_name}-frameskip={ppo_args.frame_skip}.ckpt", "wb") as f:
        pickle.dump(nnx.state(model, nnx.Param), f)
    print(f"Model saved to {model_dir}/{ppo_args.env_name}-frameskip={ppo_args.frame_skip}-notebook.ckpt")

print("Training complete!")

ENV_NAME = ppo_args.env_name
CKPT_PATH = f"{model_dir}/{ppo_args.env_name}-frameskip{ppo_args.frame_skip}.ckpt"
OUTPUT_GIF = f"{model_dir}/{ppo_args.env_name}-frameskip{ppo_args.frame_skip}.gif"
print(env.frame_skip)
rng = jax.random.PRNGKey(123)
rng, load_rng = jax.random.split(rng)

#model = load_model(env, CKPT_PATH, load_rng)
#print(f"Model loaded from {CKPT_PATH}")
#rng, eval_rng = jax.random.split(rng)
#eval_rollout_and_save_video(
#    model,
#    env,
#    rng_key=eval_rng,
#    num_envs_to_render=4,
#    max_steps=200,
#    fps=60,
#    output_gif=OUTPUT_GIF,
#)
#upload OUTPUT_GIF to wandb
#wandb.log({f"{ENV_NAME}-frameskip{ppo_args.frame_skip}": wandb.Video(OUTPUT_GIF)})

wandb.finish()