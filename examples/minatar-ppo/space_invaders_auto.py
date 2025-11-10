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
import jax
device_id = 1
jax.config.update('jax_default_device', jax.devices()[device_id])
print(f"Using device: {jax.devices()[device_id]}")
import jax.numpy as jnp


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
"""MinAtar/SpaceInvaders: A fork of github.com/kenjyoung/MinAtar

The authors of original MinAtar implementation are:
    * Kenny Young (kjyoung@ualberta.ca)
    * Tian Tian (ttian@ualberta.ca)
The original MinAtar implementation is distributed under GNU General Public License v3.0
    * https://github.com/kenjyoung/MinAtar/blob/master/License.txt
"""
from typing import Literal, Optional

import jax
import jax.lax as lax
from jax import numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

SHOT_COOL_DOWN = jnp.int32(5)
ENEMY_MOVE_INTERVAL = jnp.int32(12)
ENEMY_SHOT_INTERVAL = jnp.int32(10)

ZERO = jnp.int32(0)
NINE = jnp.int32(9)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((10, 10, 6), dtype=jnp.bool_)
    rewards: Array = jnp.zeros(1, dtype=jnp.float32)  # (1,)
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(4, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- MinAtar SpaceInvaders specific ---
    _pos: Array = jnp.int32(5)
    _f_bullet_map: Array = jnp.zeros((10, 10), dtype=jnp.bool_)
    _e_bullet_map: Array = jnp.zeros((10, 10), dtype=jnp.bool_)
    _alien_map: Array = (
        jnp.zeros((10, 10), dtype=jnp.bool_).at[0:4, 2:8].set(TRUE)
    )
    _alien_dir: Array = jnp.int32(-1)
    _enemy_move_interval: Array = ENEMY_MOVE_INTERVAL
    _alien_move_timer: Array = ENEMY_MOVE_INTERVAL
    _alien_shot_timer: Array = ENEMY_SHOT_INTERVAL
    _ramp_index: Array = jnp.int32(0)
    _shot_timer: Array = jnp.int32(0)
    _terminal: Array = FALSE
    _last_action: Array = jnp.int32(0)

    @property
    def env_id(self) -> core.EnvId:
        return "minatar-space_invaders"

    def to_svg(
        self,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> str:
        del color_theme, scale
        from .utils import visualize_minatar

        return visualize_minatar(self)

    def save_svg(
        self,
        filename,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> None:
        from .utils import visualize_minatar

        visualize_minatar(self, filename)


class MinAtarSpaceInvaders(core.Env):
    def __init__(
        self,
        *,
        use_minimal_action_set: bool = True,
        sticky_action_prob: float = 0.1,
    ):
        super().__init__()
        self.use_minimal_action_set = use_minimal_action_set
        self.sticky_action_prob: float = sticky_action_prob
        self.minimal_action_set = jnp.int32([0, 1, 3, 5])
        self.legal_action_mask = jnp.ones(6, dtype=jnp.bool_)
        if self.use_minimal_action_set:
            self.legal_action_mask = jnp.ones(
                self.minimal_action_set.shape[0], dtype=jnp.bool_
            )

    def step(
        self, state: core.State, action: Array, key: Optional[Array] = None
    ) -> core.State:
        assert key is not None, (
            "v2.0.0 changes the signature of step. Please specify PRNGKey at the third argument:\n\n"
            "  * <  v2.0.0: step(state, action)\n"
            "  * >= v2.0.0: step(state, action, key)\n\n"
            "See v2.0.0 release note for more details:\n\n"
            "  https://github.com/sotetsuk/pgx/releases/tag/v2.0.0"
        )
        return super().step(state, action, key)

    def _init(self, key: PRNGKey) -> State:
        state = State()
        state = state.replace(legal_action_mask=self.legal_action_mask)  # type: ignore
        return state  # type: ignore

    def _step(self, state: core.State, action, key) -> State:
        state = state.replace(legal_action_mask=self.legal_action_mask)  # type: ignore
        action = jax.lax.select(
            self.use_minimal_action_set,
            self.minimal_action_set[action],
            action,
        )
        return _step(state, action, key, self.sticky_action_prob)  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state)

    @property
    def id(self) -> core.EnvId:
        return "minatar-space_invaders"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self):
        return 1


def _step(
    state: State,
    action: Array,
    key,
    sticky_action_prob,
):
    action = jnp.int32(action)
    action = jax.lax.cond(
        jax.random.uniform(key) < sticky_action_prob,
        lambda: state._last_action,
        lambda: action,
    )
    return _step_det(state, action)


def _observe(state: State) -> Array:
    obs = jnp.zeros((10, 10, 6), dtype=jnp.bool_)
    obs = obs.at[9, state._pos, 0].set(TRUE)
    obs = obs.at[:, :, 1].set(state._alien_map)
    obs = obs.at[:, :, 2].set(
        lax.cond(
            state._alien_dir < 0,
            lambda: state._alien_map,
            lambda: jnp.zeros_like(state._alien_map),
        )
    )
    obs = obs.at[:, :, 3].set(
        lax.cond(
            state._alien_dir < 0,
            lambda: jnp.zeros_like(state._alien_map),
            lambda: state._alien_map,
        )
    )
    obs = obs.at[:, :, 4].set(state._f_bullet_map)
    obs = obs.at[:, :, 5].set(state._e_bullet_map)
    return obs


def _step_det(
    state: State,
    action: Array,
):
    r = jnp.float32(0)

    pos = state._pos
    f_bullet_map = state._f_bullet_map
    e_bullet_map = state._e_bullet_map
    alien_map = state._alien_map
    alien_dir = state._alien_dir
    enemy_move_interval = state._enemy_move_interval
    alien_move_timer = state._alien_move_timer
    alien_shot_timer = state._alien_shot_timer
    ramp_index = state._ramp_index
    shot_timer = state._shot_timer
    terminal = state._terminal

    # Resolve player action
    # action_map = ['n','l','u','r','d','f']
    pos, f_bullet_map, shot_timer = _resole_action(
        pos, f_bullet_map, shot_timer, action
    )

    # Update Friendly Bullets
    f_bullet_map = jnp.roll(f_bullet_map, -1, axis=0)
    f_bullet_map = f_bullet_map.at[9, :].set(FALSE)

    # Update Enemy Bullets
    e_bullet_map = jnp.roll(e_bullet_map, 1, axis=0)
    e_bullet_map = e_bullet_map.at[0, :].set(FALSE)
    terminal = lax.cond(e_bullet_map[9, pos], lambda: TRUE, lambda: terminal)

    # Update aliens
    terminal = lax.cond(alien_map[9, pos], lambda: TRUE, lambda: terminal)
    alien_move_timer, alien_map, alien_dir, terminal = lax.cond(
        alien_move_timer == 0,
        lambda: _update_alien_by_move_timer(
            alien_map, alien_dir, enemy_move_interval, pos, terminal
        ),
        lambda: (alien_move_timer, alien_map, alien_dir, terminal),
    )
    timer_zero = alien_shot_timer == 0
    alien_shot_timer = lax.cond(
        timer_zero, lambda: ENEMY_SHOT_INTERVAL, lambda: alien_shot_timer
    )
    e_bullet_map = lax.cond(
        timer_zero,
        lambda: e_bullet_map.at[_nearest_alien(pos, alien_map)].set(TRUE),
        lambda: e_bullet_map,
    )

    kill_locations = alien_map & (alien_map == f_bullet_map)

    r += jnp.sum(kill_locations, dtype=jnp.float32)
    alien_map = alien_map & (~kill_locations)
    f_bullet_map = f_bullet_map & (~kill_locations)

    # Update various timers
    shot_timer -= shot_timer > 0
    alien_move_timer -= 1
    alien_shot_timer -= 1
    ramping = True
    is_enemy_zero = jnp.count_nonzero(alien_map) == 0
    enemy_move_interval, ramp_index = lax.cond(
        is_enemy_zero & (enemy_move_interval > 6) & ramping,
        lambda: (enemy_move_interval - 1, ramp_index + 1),
        lambda: (enemy_move_interval, ramp_index),
    )
    alien_map = lax.cond(
        is_enemy_zero,
        lambda: alien_map.at[0:4, 2:8].set(TRUE),
        lambda: alien_map,
    )

    return state.replace(  # type: ignore
        _pos=pos,
        _f_bullet_map=f_bullet_map,
        _e_bullet_map=e_bullet_map,
        _alien_map=alien_map,
        _alien_dir=alien_dir,
        _enemy_move_interval=enemy_move_interval,
        _alien_move_timer=alien_move_timer,
        _alien_shot_timer=alien_shot_timer,
        _ramp_index=ramp_index,
        _shot_timer=shot_timer,
        _terminal=terminal,
        _last_action=action,
        rewards=r[jnp.newaxis],
        terminated=terminal,
    )


def _resole_action(pos, f_bullet_map, shot_timer, action):
    f_bullet_map = lax.cond(
        (action == 5) & (shot_timer == 0),
        lambda: f_bullet_map.at[9, pos].set(TRUE),
        lambda: f_bullet_map,
    )
    shot_timer = lax.cond(
        (action == 5) & (shot_timer == 0),
        lambda: SHOT_COOL_DOWN,
        lambda: shot_timer,
    )
    pos = lax.cond(
        action == 1, lambda: jax.lax.max(ZERO, pos - 1), lambda: pos
    )
    pos = lax.cond(
        action == 3, lambda: jax.lax.min(NINE, pos + 1), lambda: pos
    )
    return pos, f_bullet_map, shot_timer


def _nearest_alien(pos, alien_map):
    search_order = jnp.argsort(jnp.abs(jnp.arange(10, dtype=jnp.int32) - pos))
    ix = lax.while_loop(
        lambda i: jnp.sum(alien_map[:, search_order[i]]) <= 0,
        lambda i: i + 1,
        0,
    )
    ix = search_order[ix]
    j = lax.while_loop(lambda i: alien_map[i, ix] == 0, lambda i: i - 1, 9)
    return (j, ix)


def _update_alien_by_move_timer(
    alien_map, alien_dir, enemy_move_interval, pos, terminal
):
    alien_move_timer = lax.min(
        jnp.sum(alien_map, dtype=jnp.int32), enemy_move_interval
    )
    cond = ((jnp.sum(alien_map[:, 0]) > 0) & (alien_dir < 0)) | (
        (jnp.sum(alien_map[:, 9]) > 0) & (alien_dir > 0)
    )
    terminal = lax.cond(
        cond & (jnp.sum(alien_map[9, :]) > 0),
        lambda: jnp.bool_(True),
        lambda: terminal,
    )
    alien_dir = lax.cond(cond, lambda: -alien_dir, lambda: alien_dir)
    alien_map = lax.cond(
        cond,
        lambda: jnp.roll(alien_map, 1, axis=0),
        lambda: jnp.roll(alien_map, alien_dir, axis=1),
    )
    terminal = lax.cond(
        alien_map[9, pos], lambda: jnp.bool_(True), lambda: terminal
    )
    return alien_move_timer, alien_map, alien_dir, terminal
# -----------------------------
# Config
# -----------------------------
from pydantic import BaseModel
import pgx
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
    plan_horizon: int = 4
    total_timesteps: int = 20_000_000
    frame_skip: int = 0
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
    

    class Config:
        extra = "forbid"


# In Jupyter, directly create the config instead of parsing CLI args
# You can override any default values here
args = PPOConfig(
    env_name="minatar-space_invaders",  # Change this to test different games
    num_envs=4096,  # Smaller for testing in notebook
    total_timesteps=200_000_000,  # Shorter for testing
    plan_horizon=1,
    save_model=True,  # Don't save in notebook by default
)
print(f"Config: {args}")

env = pgx.make(str(args.env_name))
if args.env_name == "minatar-space_invaders":
    env = MinAtarSpaceInvaders()

num_updates = args.total_timesteps // args.num_envs // args.num_steps
num_minibatches = args.num_envs * args.num_steps // args.minibatch_size
ppo_args = args

# === Macro-PPO with Autoregressive (AR) Plan Head ===
import os, io, math, time, pickle, sys
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import wandb

import pgx
from pgx.experimental import auto_reset
from pydantic import BaseModel


# ---------------------------------------
# Simple categorical wrapper
# ---------------------------------------
class Categorical:
    def __init__(self, logits):
        self.logits = logits  # [..., A]

    def sample(self, seed):
        # vectorized sample across leading dims
        return jax.random.categorical(seed, self.logits, axis=-1)

    def log_prob(self, value):
        # value: shape logits.shape[:-1]
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return jnp.take_along_axis(log_probs, value[..., None], axis=-1).squeeze(-1)

    def entropy(self):
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        probs = jax.nn.softmax(self.logits, axis=-1)
        return -(probs * log_probs).sum(axis=-1)

# ---------------------------------------
# Utility
# ---------------------------------------
def pool_out_dim(n: int, window: int = 2, stride: int = 2, padding: str = "VALID") -> int:
    if padding.upper() == "VALID":
        return (n - window) // stride + 1
    return math.ceil(n / stride)

def _save_ckpt(model, model_dir, env_name, frame_skip, plan_horizon, tag):
    os.makedirs(model_dir, exist_ok=True)
    path = f"{model_dir}/{tag}.ckpt"
    with open(path, "wb") as f:
        pickle.dump(nnx.state(model, nnx.Param), f)
    print(f"[ckpt] saved: {path}")
    return path


# ---------------------------------------
# Actor-Critic with AR plan head (Macro-PPO)
# ---------------------------------------
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
        self.torso_dim = 64
        self.fc = nnx.Linear(flatten_dim, self.torso_dim, rngs=rngs)

        # single-step actor (kept for compatibility)
        self.actor_h1 = nnx.Linear(64, 64, rngs=rngs)
        self.actor_h2 = nnx.Linear(64, 64, rngs=rngs)
        self.actor_out = nnx.Linear(64, num_actions, rngs=rngs)

        # critic
        self.critic_h1 = nnx.Linear(64, 64, rngs=rngs)
        self.critic_h2 = nnx.Linear(64, 64, rngs=rngs)
        self.critic_out = nnx.Linear(64, 1, rngs=rngs)

        # AR plan head
        in_dim = self.num_actions + self.time_embed_dim + self.torso_dim

        self.hid_init = nnx.Linear(self.torso_dim, self.ar_hidden, rngs=rngs)
        self.gru_z = nnx.Linear(self.ar_hidden + in_dim, self.ar_hidden, rngs=rngs)
        self.gru_r = nnx.Linear(self.ar_hidden + in_dim, self.ar_hidden, rngs=rngs)
        self.gru_n = nnx.Linear(self.ar_hidden + in_dim, self.ar_hidden, rngs=rngs)
        self.ar_out = nnx.Linear(self.ar_hidden, self.num_actions, rngs=rngs)

    # helpers
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

    # standard API (single-step)
    def __call__(self, x):
        h = self._torso(x)
        a = self._act(self.actor_h1(h)); a = self._act(self.actor_h2(a))
        logits = self.actor_out(a)
        v = self._act(self.critic_h1(h)); v = self._act(self.critic_h2(v))
        value = self.critic_out(v)
        return logits, jnp.squeeze(value, axis=-1)

    # AR cell
    def _gru(self, h, x):
        hx = jnp.concatenate([h, x], axis=-1)
        z = jax.nn.sigmoid(self.gru_z(hx))
        r = jax.nn.sigmoid(self.gru_r(hx))
        hx_r = jnp.concatenate([r * h, x], axis=-1)
        n = jnp.tanh(self.gru_n(hx_r))
        return (1.0 - z) * h + z * n

    def _time_feat(self, t):
    # t: scalar jnp.int32/jnp.int64, works under JIT/scan
        D = self.time_embed_dim
        i = jnp.arange(D // 2, dtype=jnp.float32)                      # [D/2]
        # Use jnp.log and keep everything in JAX space
        factor = jnp.exp((-jnp.log(10000.0) * 2.0 / D) * i)            # [D/2]
        tt = jnp.asarray(t, dtype=jnp.float32)                         # scalar
        s = jnp.sin(tt * factor)                                       # [D/2]
        c = jnp.cos(tt * factor)                                       # [D/2]
        return jnp.concatenate([s, c], axis=0)                         # [D]


    # sample plan (returns actions [B,N], joint logp [B], entropy sum [B], V(o0) [B])
    def sample_plan(self, obs, rng):
        B = obs.shape[0]
        base = self._torso(obs)  # [B,64]
        v0 = jnp.squeeze(self.critic_out(self._act(self.critic_h2(self._act(self.critic_h1(base))))), -1)
        h = jnp.tanh(self.hid_init(base))  # [B,H]

        def step(carry, t):
            h, rng_in = carry
            logits = self.ar_out(h)  # [B,A]
            cat = Categorical(logits)
            rng_in, sub = jax.random.split(rng_in)
            a_t = jax.random.categorical(sub, logits)  # [B]
            logp_t = cat.log_prob(a_t)                 # [B]
            ent_t  = cat.entropy()                     # [B]
            a_oh = jax.nn.one_hot(a_t, self.num_actions, dtype=jnp.float32)
            tf = jnp.broadcast_to(self._time_feat(t), (B, self.time_embed_dim))
            x_in = jnp.concatenate([a_oh, tf, base], axis=-1)
            h_next = self._gru(h, x_in)
            return (h_next, rng_in), (a_t, logp_t, ent_t)

        (h_final, _), (A, LP, EN) = jax.lax.scan(step, (h, rng), jnp.arange(self.plan_horizon))
        actions = jnp.swapaxes(A, 0, 1)          # [B,N]
        logp_sum = jnp.swapaxes(LP, 0, 1).sum(axis=1)  # [B]
        ent_sum  = jnp.swapaxes(EN, 0, 1).sum(axis=1)  # [B]
        return actions, logp_sum, ent_sum, v0

    # teacher-forced joint log-prob for stored plans
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
            x_in = jnp.concatenate([a_oh, tf, base], axis=-1)
            return self._gru(h, x_in), (logp_t, ent_t)
        h_final, (LP, EN) = jax.lax.scan(step, h, jnp.arange(self.plan_horizon))
        return LP.sum(axis=0), EN.sum(axis=0)  # [B], [B]

# ---------------------------------------
# Macro transition
# ---------------------------------------
class MacroTransition(NamedTuple):
    done: jnp.ndarray      # [B]
    actions: jnp.ndarray   # [B,N]
    value0: jnp.ndarray    # [B]
    reward: jnp.ndarray    # [B]
    log_prob: jnp.ndarray  # [B]
    obs0: jnp.ndarray      # [B,H,W,C]


# ---------------------------------------
# Env setup (use custom MinAtarBreakout if present)
# ---------------------------------------
if ppo_args.env_name == "minatar-breakout" and "MinAtarBreakout" in globals():
    env = MinAtarBreakout(
        use_minimal_action_set=True,
        sticky_action_prob=0.1,
        frame_skip=ppo_args.frame_skip,
    )
else:
    env = pgx.make(str(ppo_args.env_name))

# ---------------------------------------
# Optimizer
# ---------------------------------------
tx = optax.chain(
    optax.clip_by_global_norm(ppo_args.max_grad_norm),
    optax.adam(ppo_args.lr, eps=1e-5),
)

# ---------------------------------------
# Update step (Macro-PPO with AR plan head)
# ---------------------------------------
def make_update_step():
    step_fn = jax.vmap(auto_reset(env.step, env.init))
    N = int(ppo_args.plan_horizon)
    gammaN = jnp.float32(ppo_args.gamma) ** N

    def apply_open_loop_plan(env_state, plan_actions, rng):
        """Execute N external env steps open-loop."""
        B = env_state.observation.shape[0]
        R = jnp.zeros((B,), dtype=jnp.float32)
        done_any = jnp.zeros((B,), dtype=jnp.bool_)

        def body(i, carry):
            state, R, done_any, rng = carry
            rng, sub = jax.random.split(rng)
            keys = jax.random.split(sub, B)
            state_next = step_fn(state, plan_actions[:, i], keys)
            r_i = jnp.squeeze(state_next.rewards, -1)      # [B]
            term_i = state_next.terminated                 # [B]
            mask_prev = 1.0 - done_any.astype(jnp.float32) # stop accumulating after first term
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
        # ---- Collect macros ----
        def _macro_step(runner_state, _):
            model, optimizer, env_state, last_obs, rng = runner_state

            # sample plan under old params
            rng, sub = jax.random.split(rng)
            actions, joint_logp, _ent, value0 = model.sample_plan(last_obs, sub)  # [B,N], [B], [B], [B]

            # roll out N steps open-loop
            rng, sub = jax.random.split(rng)
            env_state, R_chunk, done_any, _ = apply_open_loop_plan(env_state, actions, sub)

            transition = MacroTransition(
                done=done_any,
                actions=actions,
                value0=value0,
                reward=R_chunk,
                log_prob=joint_logp,
                obs0=last_obs,
            )
            runner_state = (model, optimizer, env_state, env_state.observation, rng)
            return runner_state, transition

        runner_state = (model, optimizer, env_state, last_obs, rng)
        runner_state, traj_batch = nnx.scan(
            _macro_step, in_axes=(nnx.Carry, None), out_axes=(nnx.Carry, 0), length=ppo_args.num_steps
        )(runner_state, None)

        # ---- Macro-GAE (gamma^N) ----
        model, optimizer, env_state, last_obs, rng = runner_state
        _, last_val = model(last_obs)  # [B]

        def _gae(carry, tr: MacroTransition):
            gae, next_value = carry
            delta = tr.reward + gammaN * next_value * (1 - tr.done) - tr.value0
            gae = delta + gammaN * ppo_args.gae_lambda * (1 - tr.done) * gae
            return (gae, tr.value0), gae

        (_, _), advantages = jax.lax.scan(_gae, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
        targets = advantages + traj_batch.value0  # [T,B]

        # ---- SGD epochs ----
        def _update_epoch(update_state, _):
            model, optimizer, traj_batch, advantages, targets, rng = update_state

            def _update_minibatch(state, minibatch):
                model, optimizer = state
                mb_traj, mb_adv, mb_targets = minibatch

                def _loss_fn(model: nnx.Module, traj: MacroTransition, gae, targets):
                    # joint log-prob under current params (teacher forcing)
                    new_joint_logp, ent_sum = model.plan_logprob(traj.obs0, traj.actions)  # [MB], [MB]
                    # critic at o0
                    _, value = model(traj.obs0)  # [MB]

                    # value loss (clipped)
                    value_pred_clipped = traj.value0 + (value - traj.value0).clip(-ppo_args.clip_eps, ppo_args.clip_eps)
                    v_loss_unclipped = jnp.square(value - targets)
                    v_loss_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

                    # policy loss with joint ratio
                    ratio = jnp.exp(new_joint_logp - traj.log_prob)  # [MB]
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

                optimizer.update(model=model, grads=grads)
                return (model, optimizer), (total_loss, aux)

            # flatten (T,B) -> (T*B)
            rng, _rng = jax.random.split(rng)
            batch_size = ppo_args.minibatch_size * num_minibatches
            assert batch_size == ppo_args.num_steps * ppo_args.num_envs, "batch size must equal (num_macros * num_envs)"
            batch = (traj_batch, advantages, targets)
            batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            perm = jax.random.permutation(_rng, batch_size)
            shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), batch)
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                shuffled,
            )

            (model, optimizer), losses = nnx.scan(_update_minibatch, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
                (model, optimizer), minibatches
            )
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

# ---------------------------------------
# Evaluation with open-loop plans (Macro)
# ---------------------------------------
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
        # sample a plan
        rng_key, sub = jax.random.split(rng_key)
        actions, _, _, _ = model.sample_plan(state.observation, sub)  # [B,N]
        # execute N steps
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

# ---------------------------------------
# Training loop
# ---------------------------------------
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
        # --- checkpoint schedule (4 evenly spaced by env steps) ---
    total_steps_budget = int(ppo_args.total_timesteps)
    # thresholds in *env-steps* (not updates). We’ll compare against `steps` counter.
    milestones = [int(total_steps_budget * frac) for frac in (0.25, 0.50, 0.75, 1.00)]
    saved_flags = [False, False, False, False]

    # model + optimizer
    rng, _rng = jax.random.split(rng)
    model = build_model(env, _rng)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # update fn
    update_step = make_update_step()

    # init envs
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, ppo_args.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(reset_rng)
    last_obs = env_state.observation
    rng, _rng = jax.random.split(rng)
    runner_state = (model, optimizer, env_state, last_obs, _rng)

    # warmup compile
    _, _ = update_step(*runner_state)

    # initial eval
    steps = 0
    rng, _rng = jax.random.split(rng)
    eval_R = evaluate_macro(runner_state[0], _rng)
    log = {"sec": 0.0, f"{ppo_args.env_name}/eval_R": float(eval_R), "steps": steps}
    print(log)
    if wandb.run is not None: wandb.log(log)
    st = time.time(); tt = 0.0

    for _ in range(num_updates):
        runner_state, loss_info = update_step(*runner_state)
        model, optimizer, env_state, last_obs, rng = runner_state
        steps += ppo_args.num_envs * ppo_args.num_steps * ppo_args.plan_horizon  # env steps progressed
        # --- checkpointing: save at milestones when crossed ---
        if ppo_args.save_model:
            for i, ms in enumerate(milestones):
                if not saved_flags[i] and steps >= ms:
                    _ = _save_ckpt(
                        model=runner_state[0],
                        model_dir=f"./minatar-ppo-models/{ppo_args.env_name}/{ppo_args.plan_horizon}",
                        env_name=ppo_args.env_name,
                        frame_skip=ppo_args.frame_skip,
                        plan_horizon=ppo_args.plan_horizon,
                        tag=f"{ms//1000}k"  # e.g., 250k, 500k, ...
                    )
                    saved_flags[i] = True
        et = time.time(); tt += et - st
        rng, _rng = jax.random.split(rng)
        eval_R = evaluate_macro(model, _rng)
        log = {"sec": tt, f"{ppo_args.env_name}/eval_R": float(eval_R), "steps": steps}
        print(log)
        if wandb.run is not None: wandb.log(log)
        st = time.time()

    return runner_state

# ---------------------------------------
# Bookkeeping, updates, and run
# ---------------------------------------
# compute updates from ENV-STEP budget
num_updates = ppo_args.total_timesteps // (ppo_args.num_envs * ppo_args.num_steps * ppo_args.plan_horizon)
num_minibatches = (ppo_args.num_envs * ppo_args.num_steps) // ppo_args.minibatch_size

wandb.init(
    project=f"ppo-{ppo_args.env_name}-frameskip",
    name=f"{ppo_args.env_name}-frameskip{ppo_args.frame_skip}-N{ppo_args.plan_horizon}",
    config=ppo_args.dict(),
    mode="disabled",  # set to "online" if you want logging
)

print("Starting training (Macro-PPO, AR head)...")
rng = jax.random.PRNGKey(ppo_args.seed)
runner_state = train(rng)

# save
model_dir = f"./minatar-ppo-models/{ppo_args.env_name}/{ppo_args.plan_horizon}"
os.makedirs(model_dir, exist_ok=True)

ckpt = ''
if ppo_args.save_model:
    model = runner_state[0]
    ckpt = f"{model_dir}/full.ckpt"
    with open(ckpt, "wb") as f:
        pickle.dump(nnx.state(model, nnx.Param), f)
    print(f"Model saved to {ckpt}")

wandb.finish()
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

@nnx.jit
def evaluate_macro(model: nnx.Module, rng_key):
    step_fn = jax.vmap(env.step)

    B  = 4  # or int(ppo_args.num_eval_envs)
    N  = int(getattr(model, "plan_horizon", ppo_args.plan_horizon))  # robust to ckpt
    Tm = 400  # number of macro-steps

    rng_key, sub_key = jax.random.split(rng_key)
    reset_keys = jax.random.split(sub_key, B)
    state = jax.vmap(env.init)(reset_keys)
    R = jnp.zeros_like(state.rewards)  # [B,1]

    def _repeat_state(s):
        return jax.tree.map(lambda x: jnp.broadcast_to(x[None], (N,) + x.shape), s)  # [N,...]

    def _do_macro(carry):
        s, Racc, rng = carry
        rng, sub = jax.random.split(rng)
        actions, _, _, _ = model.sample_plan(s.observation, sub)  # [B, N]

        # N micro-steps; collect states per micro-step
        def micro_step(c, a_t):
            s2, R2, r2 = c
            r2, sub2 = jax.random.split(r2)
            keys = jax.random.split(sub2, B)
            s3 = step_fn(s2, a_t, keys)
            R2 = R2 + s3.rewards
            return (s3, R2, r2), s3  # y = state at this micro-step

        (s, Racc, rng), state_seq = jax.lax.scan(micro_step, (s, Racc, rng), actions.T)  # actions.T: [N, B]
        return (s, Racc, rng), (state_seq, actions)  # y: ([N,...], [B,N])

    def _skip(c):
        s, Racc, rng = c
        dummy_actions = jnp.full((B, N), -1, dtype=jnp.int32)  # sentinel for "no plan"
        return (s, Racc, rng), (_repeat_state(s), dummy_actions)

    def macro_step(carry, _):
        s, Racc, rng = carry
        done_all = jnp.all(s.terminated)
        return jax.lax.cond(done_all, _skip, _do_macro, carry)

    # y = (states_nested, plans_nested)
    (state, R, _), (states_nested, plans_nested) = jax.lax.scan(
        macro_step, (state, R, rng_key), None, length=Tm
    )
    # states_nested leaves: [Tm, N, B, ...]
    states_ts = jax.tree.map(lambda x: x.reshape((Tm * N,) + x.shape[2:]), states_nested)

    # plans_nested: [Tm, B, N] → tile over N micro-steps → [Tm, N, B, N] → flatten time
    plans_tiled = jnp.broadcast_to(plans_nested[:, None, :, :], (Tm, N, B, N))
    plan_actions_ts = plans_tiled.reshape((Tm * N, B, N))  # each state gets its macro's plan

    return states_ts, plan_actions_ts, R.mean()
from flax import nnx
import jax
import jax.numpy as jnp
from pgx.experimental import auto_reset  # unused now, but you can keep the import

@nnx.jit(
    static_argnames=(
        "ticks", "decision_rate", "act_delay", "obs_delay",
        "num_envs", "noop_id", "fifo_capacity",
    ),
    donate_argnames=("model",)
)
def simulate_realtime(model: nnx.Module,
                      rng_key,
                      *,
                      ticks: int,
                      decision_rate: int | None = None,   # default: model.plan_horizon
                      act_delay: int = 0,                 # inference latency (ticks)
                      obs_delay: int = 0,                 # sensor latency (ticks)
                      num_envs: int | None = None,        # default: ppo_args.num_eval_envs
                      noop_id: int = 0,                   # action to repeat when FIFO empty
                      fifo_capacity: int | None = None):  # override FIFO size if you like
    """
    Real-time rollout WITHOUT auto-reset. We record the first episode return per env.

    Timing:
      - Decide every K ticks (K = decision_rate or model.plan_horizon).
      - Push matured plan BEFORE popping this tick.
      - act_delay == 0 => the plan decided this tick is matured immediately (a0 can apply now).
      - act_delay  > 0 => plan matures exactly after 'act_delay' ticks.
      - Pop exactly one action per tick per unfinished env; if empty, reuse last_action (NOOP initially).
      - Finished envs are frozen: no FIFO pops/advances, no last_action updates, and state stops changing.

    Returns:
      states_ts:        State PyTree with leading [T, B, ...]
      actions_ts:       int32 [T, B]      (applied action each tick; frozen after done)
      exec_plans_ts:    int32 [T, B, K]   (peek of next K FIFO actions; -1 where empty or finished)
      rewards_ts:       float32 [T, B]    (per-tick reward; zeros after done)
      ep_return:        float32 [B]       (sum of rewards until first termination per env)
      ep_finished_mask: bool [B]          (True where the env terminated at least once)
      ep_return_mean:   float32 []        (mean episodic return over finished envs; 0 if none finished)
    """
    # --- sizes ---
    B = int(num_envs if num_envs is not None else ppo_args.num_eval_envs)
    K = int(decision_rate if decision_rate is not None
            else getattr(model, "plan_horizon", ppo_args.plan_horizon))
    Q = int(fifo_capacity if fifo_capacity is not None else (2 * K + act_delay + 1))

    step_fn = jax.vmap(env.step)   # NO auto_reset

    # --- init envs ---
    rng_key, sub = jax.random.split(rng_key)
    reset_keys = jax.random.split(sub, B)
    state = jax.vmap(env.init)(reset_keys)

    # --- executor state ---
    last_action = jnp.full((B,), jnp.int32(noop_id), dtype=jnp.int32)

    fifo = jnp.full((B, Q), jnp.int32(-1))
    head = jnp.zeros((B,), jnp.int32)
    tail = jnp.zeros((B,), jnp.int32)

    delay_len  = max(act_delay, 1)
    plan_delay = jnp.full((delay_len, B, K), jnp.int32(-1))

    # Obs delay buffer (zeros = "no sensor yet")
    if obs_delay > 0:
        obs_delay_buf = jnp.zeros((obs_delay,) + state.observation.shape, state.observation.dtype)
    else:
        obs_delay_buf = jnp.zeros((1,) + state.observation.shape, state.observation.dtype)

    # Episodic accounting (first episode only)
    running_return   = jnp.zeros((B,), jnp.float32)
    finished         = jnp.zeros((B,), jnp.bool_)   # latched when first done happens
    ep_return        = jnp.zeros((B,), jnp.float32) # latched total at first done

    b_ix = jnp.arange(B, dtype=jnp.int32)
    zero_delay = (act_delay == 0)

    # ---- helpers ----
    def push_plan(fifo, tail, plan, mask):
        """Append a K-action plan to each env's FIFO where mask==True."""
        def body(i, carry):
            (fifo,) = carry
            pos = (tail + i) % Q
            cur = fifo[b_ix, pos]
            val = plan[:, i]
            new = jnp.where(mask, val, cur)
            fifo = fifo.at[b_ix, pos].set(new)
            return (fifo,)
        (fifo,) = jax.lax.fori_loop(0, K, body, (fifo,))
        tail = (tail + mask.astype(jnp.int32) * K) % Q
        return fifo, tail

    def peek_k(fifo, head, tail, K, Q):
        """Preview next K FIFO entries (or -1 if not present)."""
        idxs = (head[:, None] + jnp.arange(K, dtype=jnp.int32)[None, :]) % Q
        peek = fifo[b_ix[:, None], idxs]                    # [B, K]
        length = (tail - head) % Q                          # [B]
        valid  = (jnp.arange(K, dtype=jnp.int32)[None, :]   # [1, K]
                  < length[:, None])                        # [B, K]
        return jnp.where(valid, peek, jnp.int32(-1))

    def tree_where(mask, x_new, x_old):
        # mask shape [B]; broadcast to leading dim(s)
        return jax.tree.map(
            lambda a, b: jnp.where(mask.reshape((-1,) + (1,)*(a.ndim-1)), a, b),
            x_new, x_old
        )

    # ---- main scan ----
    def body(carry, t):
        (state, rng_key, fifo, head, tail, plan_delay,
         obs_delay_buf, last_action, running_return, finished, ep_return) = carry

        # 0) Decide a new plan every K ticks (observe delayed obs if requested)
        decide_now = (t % K) == 0
        obs_eff = obs_delay_buf[0] if obs_delay > 0 else state.observation

        def _decide(_):
            rng_key2, sub = jax.random.split(rng_key)
            plan, _, _, _ = model.sample_plan(obs_eff, sub)  # [B, K] (stochastic)
            return plan, rng_key2

        dummy_plan = jnp.full((B, K), jnp.int32(-1))
        plan_new, rng_key = jax.lax.cond(decide_now, _decide,
                                         lambda _: (dummy_plan, rng_key),
                                         operand=None)

        # 1) Matured plan for THIS tick; push BEFORE popping. Freeze finished envs.
        matured = jax.lax.select(zero_delay, plan_new, plan_delay[0])  # [B, K]
        push_mask = (~finished) & jnp.any(matured >= 0, axis=1)
        fifo, tail = push_plan(fifo, tail, matured, push_mask)

        # 2) POP one action (only for unfinished envs)
        has_item_env = (head != tail) & (~finished)
        a_from_fifo  = jnp.where(has_item_env, fifo[b_ix, head], jnp.int32(-1))
        action_t     = jnp.where(~finished, jnp.where(a_from_fifo >= 0, a_from_fifo, last_action), last_action)
        # Update last_action only for unfinished envs
        last_action  = jnp.where(~finished, action_t, last_action)

        # Step env (everyone), then freeze finished envs' state
        rng_key, sub = jax.random.split(rng_key)
        keys = jax.random.split(sub, B)
        state_next_all = step_fn(state, action_t, keys)

        # reward from next state
        r_t = jnp.squeeze(state_next_all.rewards, -1)  # [B]

        # Update episodic returns (only for unfinished envs)
        running_return = running_return + jnp.where(~finished, r_t, 0.0)

        # Detect first-time terminations on this tick
        new_done = (~finished) & state_next_all.terminated

        # Latch episodic return at first done
        ep_return = jnp.where(new_done, running_return, ep_return)
        finished  = finished | new_done

        # Freeze state for finished envs (keep previous state forever)
        state_next = tree_where(finished, state, state_next_all)

        # Advance head only where we popped
        head = (head + has_item_env.astype(jnp.int32)) % Q

        # 3) Maintain delay line only when act_delay > 0 (and we always write plan_new)
        if not zero_delay:
            plan_delay = jnp.roll(plan_delay, -1, axis=0)
            plan_delay = plan_delay.at[-1].set(plan_new)

        # 4) Obs delay buffer
        if obs_delay > 0:
            obs_delay_buf = jnp.roll(obs_delay_buf, -1, axis=0)
            obs_delay_buf = obs_delay_buf.at[-1].set(state.observation)

        # Exec preview; hide for finished envs
        exec_preview = peek_k(fifo, head, tail, K, Q)
        exec_preview = jnp.where(finished[:, None], jnp.int32(-1), exec_preview)

        carry = (state_next, rng_key, fifo, head, tail, plan_delay,
                 obs_delay_buf, last_action, running_return, finished, ep_return)
        out   = (state_next, action_t, exec_preview, r_t)
        return carry, out

    init = (state, rng_key, fifo, head, tail, plan_delay,
            obs_delay_buf, last_action, running_return, finished, ep_return)

    (_, _, _, _, _, _, _, _, running_return_f, finished_f, ep_return_f), \
    (states_ts, actions_ts, exec_plans_ts, rewards_ts) = jax.lax.scan(
        body, init, jnp.arange(ticks, dtype=jnp.int32)
    )

    # Mean episodic return over finished envs (avoid div-by-zero)
    finished_count = finished_f.sum()
    ep_return_mean = jnp.where(finished_count > 0,
                               ep_return_f.sum() / finished_count.astype(jnp.float32),
                               jnp.float32(0.0))

    return (states_ts,
            actions_ts,
            exec_plans_ts,
            rewards_ts,         # [T, B]
            ep_return_f,        # [B]
            finished_f,         # [B]
            ep_return_mean)     # []
checkpoint_models = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
from flax import nnx
import jax
import jax.numpy as jnp
from pgx.experimental import auto_reset  # unused now, but you can keep the import

@nnx.jit(
    static_argnames=(
        "ticks", "decision_rate", "act_delay", "obs_delay",
        "num_envs", "noop_id", "fifo_capacity",
    ),
    donate_argnames=("model",)
)
def simulate_realtime(model: nnx.Module,
                      rng_key,
                      *,
                      ticks: int,
                      decision_rate: int | None = None,   # default: model.plan_horizon
                      act_delay: int = 0,                 # inference latency (ticks)
                      obs_delay: int = 0,                 # sensor latency (ticks)
                      num_envs: int | None = None,        # default: ppo_args.num_eval_envs
                      noop_id: int = 0,                   # action to repeat when FIFO empty
                      fifo_capacity: int | None = None):  # override FIFO size if you like
    """
    Real-time rollout WITHOUT auto-reset. We record the first episode return per env.

    Timing:
      - Decide every K ticks (K = decision_rate or model.plan_horizon).
      - Push matured plan BEFORE popping this tick.
      - act_delay == 0 => the plan decided this tick is matured immediately (a0 can apply now).
      - act_delay  > 0 => plan matures exactly after 'act_delay' ticks.
      - Pop exactly one action per tick per unfinished env; if empty, reuse last_action (NOOP initially).
      - Finished envs are frozen: no FIFO pops/advances, no last_action updates, and state stops changing.

    Returns:
      states_ts:        State PyTree with leading [T, B, ...]
      actions_ts:       int32 [T, B]      (applied action each tick; frozen after done)
      exec_plans_ts:    int32 [T, B, K]   (peek of next K FIFO actions; -1 where empty or finished)
      rewards_ts:       float32 [T, B]    (per-tick reward; zeros after done)
      ep_return:        float32 [B]       (sum of rewards until first termination per env)
      ep_finished_mask: bool [B]          (True where the env terminated at least once)
      ep_return_mean:   float32 []        (mean episodic return over finished envs; 0 if none finished)
    """
    # --- sizes ---
    B = int(num_envs if num_envs is not None else ppo_args.num_eval_envs)
    K = int(decision_rate if decision_rate is not None
            else getattr(model, "plan_horizon", ppo_args.plan_horizon))
    Q = int(fifo_capacity if fifo_capacity is not None else (2 * K + act_delay + 1))

    step_fn = jax.vmap(env.step)   # NO auto_reset

    # --- init envs ---
    rng_key, sub = jax.random.split(rng_key)
    reset_keys = jax.random.split(sub, B)
    state = jax.vmap(env.init)(reset_keys)

    # --- executor state ---
    last_action = jnp.full((B,), jnp.int32(noop_id), dtype=jnp.int32)

    fifo = jnp.full((B, Q), jnp.int32(-1))
    head = jnp.zeros((B,), jnp.int32)
    tail = jnp.zeros((B,), jnp.int32)

    delay_len  = max(act_delay, 1)
    plan_delay = jnp.full((delay_len, B, K), jnp.int32(-1))

    # Obs delay buffer (zeros = "no sensor yet")
    if obs_delay > 0:
        obs_delay_buf = jnp.zeros((obs_delay,) + state.observation.shape, state.observation.dtype)
    else:
        obs_delay_buf = jnp.zeros((1,) + state.observation.shape, state.observation.dtype)

    # Episodic accounting (first episode only)
    running_return   = jnp.zeros((B,), jnp.float32)
    finished         = jnp.zeros((B,), jnp.bool_)   # latched when first done happens
    ep_return        = jnp.zeros((B,), jnp.float32) # latched total at first done

    b_ix = jnp.arange(B, dtype=jnp.int32)
    zero_delay = (act_delay == 0)

    # ---- helpers ----
    def push_plan(fifo, tail, plan, mask):
        """Append a K-action plan to each env's FIFO where mask==True."""
        def body(i, carry):
            (fifo,) = carry
            pos = (tail + i) % Q
            cur = fifo[b_ix, pos]
            val = plan[:, i]
            new = jnp.where(mask, val, cur)
            fifo = fifo.at[b_ix, pos].set(new)
            return (fifo,)
        (fifo,) = jax.lax.fori_loop(0, K, body, (fifo,))
        tail = (tail + mask.astype(jnp.int32) * K) % Q
        return fifo, tail

    def peek_k(fifo, head, tail, K, Q):
        """Preview next K FIFO entries (or -1 if not present)."""
        idxs = (head[:, None] + jnp.arange(K, dtype=jnp.int32)[None, :]) % Q
        peek = fifo[b_ix[:, None], idxs]                    # [B, K]
        length = (tail - head) % Q                          # [B]
        valid  = (jnp.arange(K, dtype=jnp.int32)[None, :]   # [1, K]
                  < length[:, None])                        # [B, K]
        return jnp.where(valid, peek, jnp.int32(-1))

    def tree_where(mask, x_new, x_old):
        # mask shape [B]; broadcast to leading dim(s)
        return jax.tree.map(
            lambda a, b: jnp.where(mask.reshape((-1,) + (1,)*(a.ndim-1)), a, b),
            x_new, x_old
        )

    # ---- main scan ----
    def body(carry, t):
        (state, rng_key, fifo, head, tail, plan_delay,
         obs_delay_buf, last_action, running_return, finished, ep_return) = carry

        # 0) Decide a new plan every K ticks (observe delayed obs if requested)
        decide_now = (t % K) == 0
        obs_eff = obs_delay_buf[0] if obs_delay > 0 else state.observation

        def _decide(_):
            rng_key2, sub = jax.random.split(rng_key)
            plan, _, _, _ = model.sample_plan(obs_eff, sub)  # [B, K] (stochastic)
            return plan, rng_key2

        dummy_plan = jnp.full((B, K), jnp.int32(-1))
        plan_new, rng_key = jax.lax.cond(decide_now, _decide,
                                         lambda _: (dummy_plan, rng_key),
                                         operand=None)

        # 1) Matured plan for THIS tick; push BEFORE popping. Freeze finished envs.
        matured = jax.lax.select(zero_delay, plan_new, plan_delay[0])  # [B, K]
        push_mask = (~finished) & jnp.any(matured >= 0, axis=1)
        fifo, tail = push_plan(fifo, tail, matured, push_mask)

        # 2) POP one action (only for unfinished envs)
        has_item_env = (head != tail) & (~finished)
        a_from_fifo  = jnp.where(has_item_env, fifo[b_ix, head], jnp.int32(-1))
        action_t     = jnp.where(~finished, jnp.where(a_from_fifo >= 0, a_from_fifo, last_action), last_action)
        # Update last_action only for unfinished envs
        last_action  = jnp.where(~finished, action_t, last_action)

        # Step env (everyone), then freeze finished envs' state
        rng_key, sub = jax.random.split(rng_key)
        keys = jax.random.split(sub, B)
        state_next_all = step_fn(state, action_t, keys)

        # reward from next state
        r_t = jnp.squeeze(state_next_all.rewards, -1)  # [B]

        # Update episodic returns (only for unfinished envs)
        running_return = running_return + jnp.where(~finished, r_t, 0.0)

        # Detect first-time terminations on this tick
        new_done = (~finished) & state_next_all.terminated

        # Latch episodic return at first done
        ep_return = jnp.where(new_done, running_return, ep_return)
        finished  = finished | new_done

        # Freeze state for finished envs (keep previous state forever)
        state_next = tree_where(finished, state, state_next_all)

        # Advance head only where we popped
        head = (head + has_item_env.astype(jnp.int32)) % Q

        # 3) Maintain delay line only when act_delay > 0 (and we always write plan_new)
        if not zero_delay:
            plan_delay = jnp.roll(plan_delay, -1, axis=0)
            plan_delay = plan_delay.at[-1].set(plan_new)

        # 4) Obs delay buffer
        if obs_delay > 0:
            obs_delay_buf = jnp.roll(obs_delay_buf, -1, axis=0)
            obs_delay_buf = obs_delay_buf.at[-1].set(state.observation)

        # Exec preview; hide for finished envs
        exec_preview = peek_k(fifo, head, tail, K, Q)
        exec_preview = jnp.where(finished[:, None], jnp.int32(-1), exec_preview)

        carry = (state_next, rng_key, fifo, head, tail, plan_delay,
                 obs_delay_buf, last_action, running_return, finished, ep_return)
        out   = (state_next, action_t, exec_preview, r_t)
        return carry, out

    init = (state, rng_key, fifo, head, tail, plan_delay,
            obs_delay_buf, last_action, running_return, finished, ep_return)

    (_, _, _, _, _, _, _, _, running_return_f, finished_f, ep_return_f), \
    (states_ts, actions_ts, exec_plans_ts, rewards_ts) = jax.lax.scan(
        body, init, jnp.arange(ticks, dtype=jnp.int32)
    )

    # Mean episodic return over finished envs (avoid div-by-zero)
    finished_count = finished_f.sum()
    ep_return_mean = jnp.where(finished_count > 0,
                               ep_return_f.sum() / finished_count.astype(jnp.float32),
                               jnp.float32(0.0))

    return (states_ts,
            actions_ts,
            exec_plans_ts,
            rewards_ts,         # [T, B]
            ep_return_f,        # [B]
            finished_f,         # [B]
            ep_return_mean)     # []
import matplotlib.pyplot as plt
import numpy as np
import re

# --- pick one: build a fresh model or load your checkpoint ---
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

# Store results: {checkpoint: {delay: mean_return}}
checkpoint_delay_results = {}

# Act delays to sweep over
act_delays = [0, 1, 2, 3, 4, 5, 6, 7, 8]

for chkp in checkpoint_models:
    model_path = f"{model_dir}/{chkp}"
    model = load_model(env, model_path, init_rng)
    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint: {chkp}")
    print(f"{'='*60}")
    
    checkpoint_delay_results[chkp] = {}
    
    for act_delay in act_delays:
        print(f"\n  Testing with act_delay={act_delay}...")
        
        # --- quick compile warmup (optional but speeds up first real call) ---
        _ = simulate_realtime(
            model, rng,
            ticks=1,
            decision_rate=ppo_args.plan_horizon,   # K
            act_delay=act_delay,
            obs_delay=0,
            num_envs=16,
            noop_id=0
        )
        
        # --- run a short real-time rollout ---
        rng, run_rng = jax.random.split(rng)
        TICKS = 1600 * ppo_args.plan_horizon
        (states_ts,
        actions_ts,
        exec_plans_ts,
        rewards_ts,
        ep_return,        # [B]
        ep_finished,      # [B]
        ep_return_mean) = simulate_realtime(
            model, run_rng,
            ticks=TICKS,
            decision_rate=ppo_args.plan_horizon,   # K=4
            act_delay=act_delay,
            obs_delay=0,       # no sensor lag
            num_envs=16,
            noop_id=0
        )
        
        mean_return = float(jax.device_get(ep_return_mean))
        checkpoint_delay_results[chkp][act_delay] = mean_return
        
        print(f"    Mean episodic return: {mean_return:.2f}")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for chkp, delays in checkpoint_delay_results.items():
    print(f"\n{chkp}:")
    for delay, return_val in delays.items():
        print(f"  delay={delay:2d}: {return_val:.2f}")

# Save results to text file
root = f"/home/ubuntu/tensorflow_test/control/real-timeRL/realtime-atari-jax/examples/minatar-ppo/{ppo_args.plan_horizon}-delay-results"

output_filename = f"{root}/{ppo_args.plan_horizon}_act_delay_chkpt_results.txt"
with open(output_filename, 'w') as f:
    f.write("="*70 + "\n")
    f.write(f"Action Delay Evaluation Results (Plan Horizon K={ppo_args.plan_horizon})\n")
    f.write("="*70 + "\n\n")
    
    # Sort checkpoints for consistent output
    def get_sort_key(checkpoint_name):
        if checkpoint_name == 'full.ckpt':
            return float('inf')
        match = re.search(r'(\d+)k?\.ckpt', checkpoint_name)
        if match:
            return int(match.group(1))
        return 0
    
    sorted_checkpoint_names = sorted(checkpoint_delay_results.keys(), key=get_sort_key)
    
    # Write header row
    f.write(f"{'Checkpoint':<20}")
    for delay in act_delays:
        f.write(f"Delay={delay:<8}")
    f.write("\n")
    f.write("-"*70 + "\n")
    
    # Write data rows
    for chkp in sorted_checkpoint_names:
        f.write(f"{chkp:<20}")
        for delay in act_delays:
            return_val = checkpoint_delay_results[chkp][delay]
            f.write(f"{return_val:<13.2f}")
        f.write("\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("Detailed Results by Checkpoint:\n")
    f.write("="*70 + "\n\n")
    
    # Write detailed results
    for chkp in sorted_checkpoint_names:
        f.write(f"{chkp}:\n")
        for delay in act_delays:
            return_val = checkpoint_delay_results[chkp][delay]
            f.write(f"  Action Delay {delay:2d}: {return_val:7.2f}\n")
        f.write("\n")

print(f"\nResults saved to: {output_filename}")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# Set seaborn style for beautiful plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12

# Sort checkpoints by training steps
def get_sort_key(checkpoint_name):
    if checkpoint_name == 'full.ckpt':
        return float('inf')  # Put 'full' at the end
    # Extract number from checkpoint name (e.g., '100000k.ckpt' -> 100000)
    match = re.search(r'(\d+)k?\.ckpt', checkpoint_name)
    if match:
        return int(match.group(1))
    return 0

# Sort checkpoints
sorted_checkpoint_names = sorted(checkpoint_delay_results.keys(), key=get_sort_key)

# Prepare data for grouped bar chart
x = np.arange(len(sorted_checkpoint_names))  # Label locations for checkpoints
width = 0.8 / len(act_delays)  # Width of bars

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Use a single base color with varying opacity
base_color = sns.color_palette("deep")[0]  # Nice blue color
alphas = np.linspace(0.3, 1.0, len(act_delays))  # Opacity from light to dark

# Plot bars for each delay
for i, delay in enumerate(act_delays):
    returns = [checkpoint_delay_results[chkp][delay] for chkp in sorted_checkpoint_names]
    offset = width * (i - len(act_delays)/2 + 0.5)
    bars = ax.bar(x + offset, returns, width, label=f'Delay {delay}', 
                   color=base_color, alpha=alphas[i], edgecolor='black', linewidth=0.7)

# Customize plot
ax.set_xlabel('Checkpoint', fontsize=16)
ax.set_ylabel('Mean Episode Return', fontsize=16)
ax.set_title('Performance Degradation with Action Delay Across Checkpoints', 
             fontsize=18, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(sorted_checkpoint_names, rotation=30, ha='right')
ax.legend(title='Action Delay', loc='upper left', framealpha=0.95, ncol=1)
#ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Add subtle spine styling
sns.despine(left=False, bottom=False)

plt.tight_layout()
root = f"/home/ubuntu/tensorflow_test/control/real-timeRL/realtime-atari-jax/examples/minatar-ppo/{ppo_args.plan_horizon}-delay-results"

output_filename_pdf = f"{root}/{ppo_args.plan_horizon}_act_delay_chkpt_results.pdf"
plt.savefig(output_filename_pdf, dpi=300, bbox_inches='tight')
plt.show()

# Also create a line plot for clearer trend visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Use a gradient of colors for different checkpoints
colors = sns.color_palette("husl", len(sorted_checkpoint_names))

for i, chkp in enumerate(sorted_checkpoint_names):
    returns = [checkpoint_delay_results[chkp][delay] for delay in act_delays]
    ax.plot(act_delays, returns, marker='o', linewidth=3, 
            markersize=10, label=chkp, color=colors[i], alpha=0.8)

ax.set_xlabel('Action Delay (ticks)', fontsize=16)
ax.set_ylabel('Mean Episode Return', fontsize=16)
ax.set_title('Performance Degradation with Action Delay', 
             fontsize=18, pad=20)
ax.legend(title='Checkpoint', loc='best', framealpha=0.95)
#ax.grid(False, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Add subtle spine styling
sns.despine(left=False, bottom=False)

plt.tight_layout()
output_filename_pdf = f"{root}/{ppo_args.plan_horizon}_performance_act_delay_chkpt_results.pdf"
plt.savefig(output_filename_pdf, dpi=300, bbox_inches='tight')
#plt.show()
