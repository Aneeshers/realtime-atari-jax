# ============================================================
# Macro-PPO with Autoregressive (AR) Plan Head for MinAtar
# - JAX-safe time features (no Python int() on tracers)
# - Open-loop N-step plans with joint PPO ratio (Macro-PPO)
# - Works with your JIT-compatible frame-skip env wrappers
# ============================================================

import sys
import os

# -----------------------------
# Project path setup
# -----------------------------
new_path = "/home/ubuntu/tensorflow_test/control/real-timeRL/realtime-atari-jax"
if new_path not in sys.path:
    sys.path.insert(0, new_path)
os.environ["PYTHONPATH"] = f"{new_path}:{os.environ.get('PYTHONPATH', '')}"
print("Python path updated:")
print(f"sys.path includes: {new_path}")
print(f"PYTHONPATH env var: {os.environ['PYTHONPATH']}")

# -----------------------------
# Imports
# -----------------------------
import math
import time
import pickle
from functools import partial
from typing import NamedTuple, Literal

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import wandb

import pgx
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from pgx.experimental import auto_reset
from pydantic import BaseModel

# -------------------------------------------------
# Minimal rendering helpers (same as your notebook)
# -------------------------------------------------
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
    assert n_channels in (4, 6, 7, 10)
    if n_channels == 4:
        return [(0.08605633600581405, 0.23824692404212, 0.30561236308077167), (0.32927729263408284, 0.4762845556584382, 0.1837155549758328), (0.8146245329198283, 0.49548316572322215, 0.5752525936416857), (0.7587183008012618, 0.7922069335474338, 0.9543861221913403)]
    elif n_channels == 6:
        return [(0.10231025194333628, 0.13952898866828906, 0.2560120319409181), (0.10594361078604106, 0.3809739011595331, 0.27015111282899046), (0.4106130272672762, 0.48044780541672255, 0.1891154277778484), (0.7829183382530567, 0.48158303462490826, 0.48672451968362596), (0.8046168329276406, 0.6365733569301846, 0.8796578402926125), (0.7775608374378459, 0.8840392521212448, 0.9452007992345052)]
    elif n_channels == 7:
        return [(0.10419418740482515, 0.11632019220053316, 0.2327552016195138), (0.08523511613408935, 0.32661779003565533, 0.2973201282529313), (0.26538761550634205, 0.4675654910052002, 0.1908220644759285), (0.6328422475018423, 0.4747981096220677, 0.29070209208025455), (0.8306875710682655, 0.5175161303658079, 0.6628221028832032), (0.7779565181455343, 0.7069421942599752, 0.9314406084043191), (0.7964528047840354, 0.908668973545918, 0.9398253500983916)]
    elif n_channels == 10:
        return [(0.09854228363950114, 0.07115215572295082, 0.16957891809124037), (0.09159726558869188, 0.20394337960213008, 0.29623965888210324), (0.09406611799930162, 0.3578871412608098, 0.2837709711722866), (0.23627685553553793, 0.46114369021199075, 0.19770731888985724), (0.49498740849493095, 0.4799034869159042, 0.21147789468974837), (0.7354526513473981, 0.4748861903571046, 0.40254094042448907), (0.8325928529853291, 0.5253446757844744, 0.6869376931865354), (0.7936920632275369, 0.6641337211433709, 0.9042311843062529), (0.7588424692372241, 0.8253990353420474, 0.9542699331220588), (0.8385645211683802, 0.9411869386771845, 0.9357655639413166)]

def visualize_minatar(state, savefile=None, fmt="svg", dpi=160):
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
        from io import BytesIO, StringIO
        if fmt == "svg":
            sio = StringIO()
            plt.savefig(sio, format="svg", bbox_inches="tight")
            plt.close(fig)
            return sio.getvalue()
        else:
            bio = BytesIO()
            plt.savefig(bio, format=fmt, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            bio.seek(0)
            return bio.getvalue()
    else:
        plt.savefig(savefile, format=fmt, bbox_inches="tight", dpi=(None if fmt == "svg" else dpi))
        plt.close(fig)
        return savefile

# -----------------------------
# CLI / Config
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
    lr: float = 3e-4
    num_envs: int = 4096
    num_eval_envs: int = 100
    num_steps: int = 128              # number of MACROS per update
    plan_horizon: int = 4             # N (macro length)
    total_timesteps: int = 20_000_000 # counted in ENV STEPS
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

# Your custom envs with JIT-safe K-frame skipping
from breakout_frame_skip import MinAtarBreakout
from freeway_frame_skip import MinAtarFreeway

# -----------------------------
# Categorical helper
# -----------------------------
class Categorical:
    def __init__(self, logits):
        self.logits = logits  # [..., A]
    def sample(self, seed):
        return jax.random.categorical(seed, self.logits, axis=-1)
    def log_prob(self, value):
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return jnp.take_along_axis(log_probs, value[..., None], axis=-1).squeeze(-1)
    def entropy(self):
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        probs = jax.nn.softmax(self.logits, axis=-1)
        return -(probs * log_probs).sum(axis=-1)

# -----------------------------
# Utility
# -----------------------------
def pool_out_dim(n: int, window: int = 2, stride: int = 2, padding: str = "VALID") -> int:
    if padding.upper() == "VALID":
        return (n - window) // stride + 1
    return math.ceil(n / stride)

# -----------------------------
# Actor-Critic with AR plan head
# -----------------------------
class ActorCritic(nnx.Module):
    def __init__(
        self,
        num_actions: int,
        obs_shape,
        activation: str = "tanh",
        *,
        rngs: nnx.Rngs,
        plan_horizon: int = 4,
        ar_hidden: int = 128,
        time_embed_dim: int = 16,
    ):
        assert activation in ["relu", "tanh"]
        self.num_actions = int(num_actions)
        self.activation = activation
        self.plan_horizon = int(plan_horizon)
        self.ar_hidden = int(ar_hidden)
        self.time_embed_dim = int(time_embed_dim)

        H, W, C = obs_shape
        # shared torso
        self.conv = nnx.Conv(in_features=C, out_features=32, kernel_size=(2, 2), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        H2 = pool_out_dim(H, 2, 2, "VALID")
        W2 = pool_out_dim(W, 2, 2, "VALID")
        flatten_dim = H2 * W2 * 32
        self.fc = nnx.Linear(flatten_dim, 64, rngs=rngs)

        # keep single-step actor/critic for compatibility
        self.actor_h1 = nnx.Linear(64, 64, rngs=rngs)
        self.actor_h2 = nnx.Linear(64, 64, rngs=rngs)
        self.actor_out = nnx.Linear(64, num_actions, rngs=rngs)

        self.critic_h1 = nnx.Linear(64, 64, rngs=rngs)
        self.critic_h2 = nnx.Linear(64, 64, rngs=rngs)
        self.critic_out = nnx.Linear(64, 1, rngs=rngs)

        # AR head
        in_dim = self.num_actions + self.time_embed_dim
        self.hid_init = nnx.Linear(64, self.ar_hidden, rngs=rngs)
        self.gru_z = nnx.Linear(self.ar_hidden + in_dim, self.ar_hidden, rngs=rngs)
        self.gru_r = nnx.Linear(self.ar_hidden + in_dim, self.ar_hidden, rngs=rngs)
        self.gru_n = nnx.Linear(self.ar_hidden + in_dim, self.ar_hidden, rngs=rngs)
        self.ar_out = nnx.Linear(self.ar_hidden, self.num_actions, rngs=rngs)

    def _act(self, x): 
        return nnx.relu(x) if self.activation == "relu" else nnx.tanh(x)

    def _torso(self, x):
        x = x.astype(jnp.float32)
        x = self.conv(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)
        x = x.reshape((x.shape[0], -1))
        x = nnx.relu(self.fc(x))
        return x  # [B,64]

    # standard single-step (unused in macro training, kept for compatibility)
    def __call__(self, x):
        h = self._torso(x)
        a = self._act(self.actor_h1(h)); a = self._act(self.actor_h2(a))
        logits = self.actor_out(a)
        v = self._act(self.critic_h1(h)); v = self._act(self.critic_h2(v))
        value = self.critic_out(v)
        return logits, jnp.squeeze(value, axis=-1)

    # GRU cell
    def _gru(self, h, x):
        hx = jnp.concatenate([h, x], axis=-1)
        z = jax.nn.sigmoid(self.gru_z(hx))
        r = jax.nn.sigmoid(self.gru_r(hx))
        hx_r = jnp.concatenate([r * h, x], axis=-1)
        n = jnp.tanh(self.gru_n(hx_r))
        return (1.0 - z) * h + z * n

    # JAX-safe time embedding (t is a tracer)
    def _time_feat(self, t):
        D = self.time_embed_dim
        i = jnp.arange(D // 2, dtype=jnp.float32)                         # [D/2]
        factor = jnp.exp((-jnp.log(10000.0) * 2.0 / D) * i)               # [D/2]
        tt = jnp.asarray(t, dtype=jnp.float32)                            # scalar
        s = jnp.sin(tt * factor); c = jnp.cos(tt * factor)                # [D/2]
        return jnp.concatenate([s, c], axis=0)                            # [D]

    # sample AR plan: actions [B,N], joint logp [B], entropy sum [B], V(o0) [B]
    def sample_plan(self, obs, rng):
        B = obs.shape[0]
        base = self._torso(obs)  # [B,64]
        v0 = jnp.squeeze(self.critic_out(self._act(self.critic_h2(self._act(self.critic_h1(base))))), -1)
        h = jnp.tanh(self.hid_init(base))  # [B,H]

        def step(carry, t):
            h, rng_in = carry
            logits = self.ar_out(h)                 # [B,A]
            cat = Categorical(logits)
            rng_in, sub = jax.random.split(rng_in)
            a_t = jax.random.categorical(sub, logits)    # [B]
            logp_t = cat.log_prob(a_t)                   # [B]
            ent_t  = cat.entropy()                       # [B]
            a_oh = jax.nn.one_hot(a_t, self.num_actions, dtype=jnp.float32)
            tf = jnp.broadcast_to(self._time_feat(t), (B, self.time_embed_dim))
            x_in = jnp.concatenate([a_oh, tf], axis=-1)
            h_next = self._gru(h, x_in)
            return (h_next, rng_in), (a_t, logp_t, ent_t)

        (h_final, _), (A, LP, EN) = jax.lax.scan(step, (h, rng), jnp.arange(self.plan_horizon))
        actions = jnp.swapaxes(A, 0, 1)                    # [B,N]
        logp_sum = jnp.swapaxes(LP, 0, 1).sum(axis=1)      # [B]
        ent_sum  = jnp.swapaxes(EN, 0, 1).sum(axis=1)      # [B]
        return actions, logp_sum, ent_sum, v0

    # teacher-forced joint logprob for stored plans
    def plan_logprob(self, obs, actions):
        B, N = actions.shape
        assert N == self.plan_horizon
        base = self._torso(obs)
        h = jnp.tanh(self.hid_init(base))
        def step(h, t):
            logits = self.ar_out(h)
            cat = Categorical(logits)
            a_t = actions[:, t]
            logp_t = cat.log_prob(a_t)
            ent_t  = cat.entropy()
            a_oh = jax.nn.one_hot(a_t, self.num_actions, dtype=jnp.float32)
            tf = jnp.broadcast_to(self._time_feat(t), (B, self.time_embed_dim))
            x_in = jnp.concatenate([a_oh, tf], axis=-1)
            return self._gru(h, x_in), (logp_t, ent_t)
        h_final, (LP, EN) = jax.lax.scan(step, h, jnp.arange(self.plan_horizon))
        return LP.sum(axis=0), EN.sum(axis=0)  # [B], [B]

# -----------------------------
# Macro transition container
# -----------------------------
class MacroTransition(NamedTuple):
    done: jnp.ndarray      # [B]
    actions: jnp.ndarray   # [B,N]
    value0: jnp.ndarray    # [B]
    reward: jnp.ndarray    # [B]  (discounted inside-chunk)
    log_prob: jnp.ndarray  # [B]  (joint old logp)
    obs0: jnp.ndarray      # [B,H,W,C]

# -----------------------------
# Update step (Macro-PPO)
# -----------------------------
def make_update_step():
    step_fn = jax.vmap(auto_reset(env.step, env.init))
    N = int(ppo_args.plan_horizon)
    gammaN = jnp.float32(ppo_args.gamma) ** N

    def apply_open_loop_plan(env_state, plan_actions, rng):
        """Execute N external env steps open-loop (stop reward after first terminal)."""
        B = env_state.observation.shape[0]
        R = jnp.zeros((B,), dtype=jnp.float32)
        done_any = jnp.zeros((B,), dtype=jnp.bool_)

        def body(i, carry):
            state, R, done_any, rng = carry
            rng, sub = jax.random.split(rng)
            keys = jax.random.split(sub, B)
            state_next = step_fn(state, plan_actions[:, i], keys)
            r_i = jnp.squeeze(state_next.rewards, -1)            # [B]
            term_i = state_next.terminated                       # [B]
            mask_prev = 1.0 - done_any.astype(jnp.float32)
            R = R + (jnp.float32(ppo_args.gamma) ** i) * mask_prev * r_i
            done_any = jnp.logical_or(done_any, term_i)
            return (state_next, R, done_any, rng)

        state, R, done_any, rng = jax.lax.fori_loop(0, N, body, (env_state, R, done_any, rng))
        return state, R, done_any, rng

    @nnx.jit(donate_argnames=("model", "optimizer"))
    def _update_step(model: nnx.Module,
                     optimizer: nnx.Optimizer,
                     env_state,
                     last_obs,
                     rng):
        # ---- collect macros ----
        def _macro_step(runner_state, _):
            model, optimizer, env_state, last_obs, rng = runner_state

            rng, sub = jax.random.split(rng)
            actions, joint_logp, _ent, value0 = model.sample_plan(last_obs, sub)  # [B,N], [B], [B], [B]

            rng, sub = jax.random.split(rng)
            env_state, R_chunk, done_any, _ = apply_open_loop_plan(env_state, actions, sub)

            transition = MacroTransition(done_any, actions, value0, R_chunk, joint_logp, last_obs)
            runner_state = (model, optimizer, env_state, env_state.observation, rng)
            return runner_state, transition

        runner_state = (model, optimizer, env_state, last_obs, rng)
        runner_state, traj_batch = nnx.scan(
            _macro_step, in_axes=(nnx.Carry, None), out_axes=(nnx.Carry, 0), length=ppo_args.num_steps
        )(runner_state, None)

        # ---- macro GAE with gamma^N ----
        model, optimizer, env_state, last_obs, rng = runner_state
        _, last_val = model(last_obs)

        def _gae(carry, tr: MacroTransition):
            gae, next_value = carry
            delta = tr.reward + gammaN * next_value * (1 - tr.done) - tr.value0
            gae = delta + gammaN * ppo_args.gae_lambda * (1 - tr.done) * gae
            return (gae, tr.value0), gae

        (_, _), advantages = jax.lax.scan(
            _gae, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16
        )
        targets = advantages + traj_batch.value0  # [T,B]

        # ---- SGD epochs ----
        def _update_epoch(update_state, _):
            model, optimizer, traj_batch, advantages, targets, rng = update_state

            def _update_minibatch(state, minibatch):
                model, optimizer = state
                mb_traj, mb_adv, mb_targets = minibatch

                def _loss_fn(model: nnx.Module, traj: MacroTransition, gae, targets):
                    # teacher-forced joint logp under current params
                    new_joint_logp, ent_sum = model.plan_logprob(traj.obs0, traj.actions)
                    # critic at o0
                    _, value = model(traj.obs0)

                    # value loss (clipped)
                    value_pred_clipped = traj.value0 + (value - traj.value0).clip(-ppo_args.clip_eps, ppo_args.clip_eps)
                    v_loss_unclipped = jnp.square(value - targets)
                    v_loss_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

                    # policy loss with joint ratio
                    ratio = jnp.exp(new_joint_logp - traj.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - ppo_args.clip_eps, 1.0 + ppo_args.clip_eps) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                    # entropy bonus (normalize by N for scale)
                    entropy = (ent_sum / float(ppo_args.plan_horizon)).mean()

                    total = loss_actor + ppo_args.vf_coef * value_loss - ppo_args.ent_coef * entropy
                    return total, (value_loss, loss_actor, entropy)

                (total_loss, aux), grads = nnx.value_and_grad(
                    _loss_fn, has_aux=True, argnums=nnx.DiffState(0, nnx.Param)
                )(model, mb_traj, mb_adv, mb_targets)

                optimizer.update(grads=grads)
                return (model, optimizer), (total_loss, aux)

            rng, _rng = jax.random.split(rng)
            batch_size = ppo_args.minibatch_size * num_minibatches
            assert batch_size == ppo_args.num_steps * ppo_args.num_envs, \
                "batch size must equal (num_macros * num_envs)"

            batch = (traj_batch, advantages, targets)
            batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            perm = jax.random.permutation(_rng, batch_size)
            shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), batch)
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                shuffled,
            )

            (model, optimizer), losses = nnx.scan(
                _update_minibatch, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
            )((model, optimizer), minibatches)
            update_state = (model, optimizer, traj_batch, advantages, targets, rng)
            return update_state, losses

        update_state = (model, optimizer, traj_batch, advantages, targets, rng)
        update_state, loss_info = nnx.scan(
            _update_epoch, in_axes=(nnx.Carry, None), out_axes=(nnx.Carry, 0), length=ppo_args.update_epochs
        )(update_state, None)

        model, optimizer, _, _, _, rng = update_state
        runner_state = (model, optimizer, env_state, last_obs, rng)
        return runner_state, loss_info

    return _update_step

# -----------------------------
# Macro evaluation (open-loop)
# -----------------------------
@nnx.jit
def evaluate_macro(model: nnx.Module, rng_key):
    step_fn = jax.vmap(env.step)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, ppo_args.num_eval_envs)
    state = jax.vmap(env.init)(subkeys)
    R = jnp.zeros_like(state.rewards)  # [B,1]

    def cond_fn(tup):
        state, *_ = tup
        return ~state.terminated.all()

    def macro_body(tup):
        state, R, rng_key = tup
        rng_key, sub = jax.random.split(rng_key)
        actions, _, _, _ = model.sample_plan(state.observation, sub)  # [B,N]
        def step_body(i, carry):
            s, Racc, rng = carry
            rng, sub2 = jax.random.split(rng)
            keys = jax.random.split(sub2, s.observation.shape[0])
            s = step_fn(s, actions[:, i], keys)
            Racc = Racc + s.rewards
            return (s, Racc, rng)
        state, R, rng_key = jax.lax.fori_loop(0, ppo_args.plan_horizon, step_body, (state, R, rng_key))
        return state, R, rng_key

    state, R, _ = jax.lax.while_loop(cond_fn, macro_body, (state, R, rng_key))
    return R.mean()

# -----------------------------
# Build & Train
# -----------------------------
def build_model(env, rng):
    obs_shape = env.observation_shape
    return ActorCritic(
        env.num_actions,
        obs_shape=obs_shape,
        activation="tanh",
        rngs=nnx.Rngs(rng),
        plan_horizon=ppo_args.plan_horizon,
        ar_hidden=128,
        time_embed_dim=16,
    )

def train(rng):
    rng, _rng = jax.random.split(rng)
    model = build_model(env, _rng)
    # print model parameters number
    nnx.display(model)
    tx = optax.chain(
        optax.clip_by_global_norm(ppo_args.max_grad_norm),
        optax.adam(ppo_args.lr, eps=1e-5),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    update_step = make_update_step()

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, ppo_args.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(reset_rng)
    last_obs = env_state.observation

    rng, _rng = jax.random.split(rng)
    runner_state = (model, optimizer, env_state, last_obs, _rng)

    # Warmup (compile)
    _, _ = update_step(*runner_state)

    # Initial eval
    steps = 0
    rng, _rng = jax.random.split(rng)
    eval_R = evaluate_macro(runner_state[0], _rng)
    log = {"sec": 0.0, f"{ppo_args.env_name}/eval_R": float(eval_R), "steps": steps}
    print(log); 
    if wandb.run is not None: wandb.log(log)
    st = time.time(); tt = 0.0

    for _ in range(num_updates):
        runner_state, loss_info = update_step(*runner_state)
        model, optimizer, env_state, last_obs, rng = runner_state
        steps += ppo_args.num_envs * ppo_args.num_steps * ppo_args.plan_horizon  # env steps progressed

        et = time.time(); tt += et - st
        rng, _rng = jax.random.split(rng)
        eval_R = evaluate_macro(model, _rng)
        log = {"sec": tt, f"{ppo_args.env_name}/eval_R": float(eval_R), "steps": steps}
        print(log)
        if wandb.run is not None: wandb.log(log)
        st = time.time()

    return runner_state

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="minatar-freeway")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_eval_envs", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=128)              # macros/update
    parser.add_argument("--plan_horizon", type=int, default=4)             # N
    parser.add_argument("--total_timesteps", type=int, default=500_000_000) # ENV steps budget
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--update_epochs", type=int, default=3)
    parser.add_argument("--minibatch_size", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="pgx-minatar-ppo")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    
    
    if args.env_name == "minatar-breakout":
        if args.plan_horizon == 1 or args.plan_horizon == 2:
            args.total_timesteps = 500_000_000
        elif args.plan_horizon == 3:
            args.total_timesteps = 3_000_000_000
        elif args.plan_horizon == 4:
            args.total_timesteps = 10_000_000_000
        elif args.plan_horizon == 5:
            args.total_timesteps = 20_000_000_000
        elif args.plan_horizon == 6:
            args.total_timesteps = 20_000_000_000
    elif args.env_name == "minatar-freeway":
        if args.plan_horizon == 1 or args.plan_horizon == 2:
            args.total_timesteps = 300_000_000
        elif args.plan_horizon == 3:
            args.total_timesteps = 300_000_000
        elif args.plan_horizon == 4:
            args.total_timesteps = 700_000_000
        elif args.plan_horizon == 5:
            args.total_timesteps = 3_000_000_000
        elif args.plan_horizon == 6:
            args.total_timesteps = 3_000_000_000
    
    ppo_args = PPOConfig(**vars(args))
    print(f"Config: {ppo_args}")

    # Device selection (set before JAX allocations)
    print(f"Available JAX devices: {jax.devices()}")
    device_id = ppo_args.gpu_id if ppo_args.gpu_id < len(jax.devices()) else 0
    if device_id != ppo_args.gpu_id:
        print(f"Warning: GPU ID {ppo_args.gpu_id} not available. Using device 0 instead.")
    jax.config.update('jax_default_device', jax.devices()[device_id])
    print(f"Using device: {jax.devices()[device_id]}")

    # Env
    env = pgx.make(str(ppo_args.env_name))
    if ppo_args.env_name == "minatar-freeway":
        print("Using custom Freeway env with frame skip")
        env = MinAtarFreeway(use_minimal_action_set=True, sticky_action_prob=0.1, frame_skip=ppo_args.frame_skip)
    if ppo_args.env_name == "minatar-breakout":
        print("Using custom Breakout env with frame skip")
        env = MinAtarBreakout(use_minimal_action_set=True, sticky_action_prob=0.1, frame_skip=ppo_args.frame_skip)

    # Budget → num_updates/minibatches
    num_updates = ppo_args.total_timesteps // (ppo_args.num_envs * ppo_args.num_steps * ppo_args.plan_horizon)
    num_minibatches = (ppo_args.num_envs * ppo_args.num_steps) // ppo_args.minibatch_size

    wandb.init(
        project=f"ppo-{ppo_args.env_name}",
        name=f"{ppo_args.env_name}-autoregressive-fs{ppo_args.frame_skip}-N{ppo_args.plan_horizon}-135k-params",
        config=ppo_args.dict(),
        #mode="disabled",  # uncomment to disable logging
    )

    print("Starting training (Macro-PPO, AR head)...")
    rng = jax.random.PRNGKey(ppo_args.seed)
    runner_state = train(rng)

    # Save
    model_dir = f"./minatar-ppo-autoregressive-models/{ppo_args.env_name}/"
    os.makedirs(model_dir, exist_ok=True)
    if ppo_args.save_model:
        model = runner_state[0]
        ckpt = f"{model_dir}/{ppo_args.env_name}-autoregressive-fs={ppo_args.frame_skip}-N={ppo_args.plan_horizon}-135k-params.ckpt"
        with open(ckpt, "wb") as f:
            pickle.dump(nnx.state(model, nnx.Param), f)
        print(f"Model saved to {ckpt}")

    print("Training complete!")
    wandb.finish()
