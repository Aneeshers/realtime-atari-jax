"""Custom MinAtar Freeway Environment

This is an example of how to create a custom PGX environment by modifying
the original MinAtarFreeway class with custom parameters and behavior.
"""
from typing import Literal, Optional

import jax
from jax import numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

# Custom parameters - modify these to change game behavior
CUSTOM_PLAYER_SPEED = jnp.array(2, dtype=jnp.int32)  # Slower than original (3)
CUSTOM_TIME_LIMIT = jnp.array(3000, dtype=jnp.int32)  # Longer than original (2500)
CUSTOM_REWARD_SCALE = 2.0  # Double rewards

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
ZERO = jnp.array(0, dtype=jnp.int32)
ONE = jnp.array(1, dtype=jnp.int32)
NINE = jnp.array(9, dtype=jnp.int32)


@dataclass
class CustomFreewayState(core.State):
    """Custom state with additional fields for modified behavior."""
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((10, 10, 7), dtype=jnp.bool_)
    rewards: Array = jnp.zeros(1, dtype=jnp.float32)  # (1,)
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(3, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- Custom MinAtar Freeway specific ---
    _cars: Array = jnp.zeros((8, 4), dtype=jnp.int32)
    _pos: Array = jnp.array(9, dtype=jnp.int32)
    _move_timer: Array = jnp.array(CUSTOM_PLAYER_SPEED, dtype=jnp.int32)
    _terminate_timer: Array = jnp.array(CUSTOM_TIME_LIMIT, dtype=jnp.int32)
    _terminal: Array = jnp.array(False, dtype=jnp.bool_)
    _last_action: Array = jnp.array(0, dtype=jnp.int32)
    # Custom fields
    _total_rewards: Array = jnp.array(0.0, dtype=jnp.float32)  # Track total rewards
    _crossings: Array = jnp.array(0, dtype=jnp.int32)  # Track successful crossings

    @property
    def env_id(self) -> core.EnvId:
        return "custom-freeway"

    def to_svg(
        self,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> str:
        del color_theme, scale
        from pgx.minatar.utils import visualize_minatar
        return visualize_minatar(self)

    def save_svg(
        self,
        filename,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> None:
        from pgx.minatar.utils import visualize_minatar
        visualize_minatar(self, filename)


class CustomMinAtarFreeway(core.Env):
    """Custom MinAtar Freeway environment with modified parameters and behavior."""
    
    def __init__(
        self,
        *,
        use_minimal_action_set: bool = True,
        sticky_action_prob: float = 0.1,
        reward_scale: float = CUSTOM_REWARD_SCALE,
        player_speed: int = CUSTOM_PLAYER_SPEED,
        time_limit: int = CUSTOM_TIME_LIMIT,
    ):
        super().__init__()
        self.use_minimal_action_set = use_minimal_action_set
        self.sticky_action_prob: float = sticky_action_prob
        self.reward_scale: float = reward_scale
        self.player_speed = jnp.array(player_speed, dtype=jnp.int32)
        self.time_limit = jnp.array(time_limit, dtype=jnp.int32)
        
        self.minimal_action_set = jnp.int32([0, 2, 4])
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

    def _init(self, key: PRNGKey) -> CustomFreewayState:
        state = _custom_init(rng=key, player_speed=self.player_speed, time_limit=self.time_limit)
        state = state.replace(legal_action_mask=self.legal_action_mask)
        return state

    def _step(self, state: core.State, action, key) -> CustomFreewayState:
        assert isinstance(state, CustomFreewayState)
        state = state.replace(legal_action_mask=self.legal_action_mask)
        action = jax.lax.select(
            self.use_minimal_action_set,
            self.minimal_action_set[action],
            action,
        )
        return _custom_step(state, action, key, self.sticky_action_prob, 
                          self.player_speed, self.time_limit, self.reward_scale)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, CustomFreewayState)
        return _observe(state)

    @property
    def id(self) -> core.EnvId:
        return "custom-freeway"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self):
        return 1


def _custom_step(
    state: CustomFreewayState,
    action: Array,
    key,
    sticky_action_prob,
    player_speed,
    time_limit,
    reward_scale,
):
    """Custom step function with modified behavior."""
    action = jnp.int32(action)
    key0, key1 = jax.random.split(key, 2)
    action = jax.lax.cond(
        jax.random.uniform(key0) < sticky_action_prob,
        lambda: state._last_action,
        lambda: action,
    )
    speeds, directions = _random_speed_directions(key1)
    return _custom_step_det(state, action, speeds=speeds, directions=directions,
                           player_speed=player_speed, time_limit=time_limit, 
                           reward_scale=reward_scale)


def _custom_init(rng: Array, player_speed: Array, time_limit: Array) -> CustomFreewayState:
    """Custom initialization with configurable parameters."""
    speeds, directions = _random_speed_directions(rng)
    return _custom_init_det(speeds=speeds, directions=directions, 
                           player_speed=player_speed, time_limit=time_limit)


def _custom_step_det(
    state: CustomFreewayState,
    action: Array,
    speeds: Array,
    directions: Array,
    player_speed: Array,
    time_limit: Array,
    reward_scale: float,
):
    """Custom deterministic step with modified game logic."""
    cars = state._cars
    pos = state._pos
    move_timer = state._move_timer
    terminate_timer = state._terminate_timer
    terminal = state._terminal
    last_action = action
    total_rewards = state._total_rewards
    crossings = state._crossings

    r = jnp.array(0, dtype=jnp.float32)

    # Modified movement logic with custom player speed
    move_timer, pos = jax.lax.cond(
        (action == 2) & (move_timer == 0),
        lambda: (player_speed, jax.lax.max(ZERO, pos - ONE)),
        lambda: (move_timer, pos),
    )
    move_timer, pos = jax.lax.cond(
        (action == 4) & (move_timer == 0),
        lambda: (player_speed, jax.lax.min(NINE, pos + ONE)),
        lambda: (move_timer, pos),
    )

    # Modified win condition with scaled rewards
    cars, r, pos, crossings = jax.lax.cond(
        pos == 0,
        lambda: (
            _randomize_cars(speeds, directions, cars, initialize=False),
            r + reward_scale,  # Apply reward scaling
            NINE,
            crossings + 1,  # Track crossings
        ),
        lambda: (cars, r, pos, crossings),
    )

    pos, cars = _update_cars(pos, cars)

    # Update various timers
    move_timer = jax.lax.cond(
        move_timer > 0, lambda: move_timer - 1, lambda: move_timer
    )
    terminate_timer -= ONE
    terminal = terminate_timer < 0
    
    # Update total rewards
    total_rewards += r

    next_state = state.replace(
        _cars=cars,
        _pos=pos,
        _move_timer=move_timer,
        _terminate_timer=terminate_timer,
        _terminal=terminal,
        _last_action=last_action,
        _total_rewards=total_rewards,
        _crossings=crossings,
        rewards=r[jnp.newaxis],
        terminated=terminal,
    )

    return next_state


def _update_cars(pos, cars):
    """Update car positions (same as original)."""
    def _update_stopped_car(pos, car):
        car = car.at[2].set(jax.lax.abs(car[3]))
        car = jax.lax.cond(
            car[3] > 0, lambda: car.at[0].add(1), lambda: car.at[0].add(-1)
        )
        car = jax.lax.cond(car[0] < 0, lambda: car.at[0].set(9), lambda: car)
        car = jax.lax.cond(car[0] > 9, lambda: car.at[0].set(0), lambda: car)
        pos = jax.lax.cond(
            (car[0] == 4) & (car[1] == pos), lambda: NINE, lambda: pos
        )
        return pos, car

    def _update_car(pos, car):
        pos = jax.lax.cond(
            (car[0] == 4) & (car[1] == pos), lambda: NINE, lambda: pos
        )
        pos, car = jax.lax.cond(
            car[2] == 0,
            lambda: _update_stopped_car(pos, car),
            lambda: (pos, car.at[2].add(-1)),
        )
        return pos, car

    pos, cars = jax.lax.scan(_update_car, pos, cars)
    return pos, cars


def _custom_init_det(speeds: Array, directions: Array, player_speed: Array, time_limit: Array) -> CustomFreewayState:
    """Custom initialization with configurable parameters."""
    cars = _randomize_cars(speeds, directions, initialize=True)
    return CustomFreewayState(
        _cars=cars,
        _move_timer=player_speed,
        _terminate_timer=time_limit,
    )


def _randomize_cars(
    speeds: Array,
    directions: Array,
    cars: Array = jnp.zeros((8, 4), dtype=int),
    initialize: bool = False,
) -> Array:
    """Randomize car positions and speeds (same as original)."""
    speeds *= directions

    def _init(_cars):
        _cars = _cars.at[:, 1].set(jnp.arange(1, 9))
        _cars = _cars.at[:, 2].set(jax.lax.abs(speeds))
        _cars = _cars.at[:, 3].set(speeds)
        return _cars

    def _update(_cars):
        _cars = _cars.at[:, 2].set(abs(speeds))
        _cars = _cars.at[:, 3].set(speeds)
        return _cars

    return jax.lax.cond(initialize, _init, _update, cars)


def _random_speed_directions(rng):
    """Generate random speeds and directions (same as original)."""
    rng1, rng2 = jax.random.split(rng, 2)
    speeds = jax.random.randint(rng1, [8], 1, 6, dtype=jnp.int32)
    directions = jax.random.choice(
        rng2, jnp.array([-1, 1], dtype=jnp.int32), [8]
    )
    return speeds, directions


def _observe(state: CustomFreewayState) -> Array:
    """Generate observation (same as original)."""
    obs = jnp.zeros((10, 10, 7), dtype=jnp.bool_)
    obs = obs.at[state._pos, 4, 0].set(TRUE)

    def _update_obs(i, _obs):
        car = state._cars[i]
        _obs = _obs.at[car[1], car[0], 1].set(TRUE)
        back_x = jax.lax.cond(
            car[3] > 0, lambda: car[0] - 1, lambda: car[0] + 1
        )
        back_x = jax.lax.cond(back_x < 0, lambda: NINE, lambda: back_x)
        back_x = jax.lax.cond(back_x > 9, lambda: ZERO, lambda: back_x)
        trail = jax.lax.abs(car[3]) + 1
        _obs = _obs.at[car[1], back_x, trail].set(TRUE)
        return _obs

    obs = jax.lax.fori_loop(0, 8, _update_obs, obs)
    return obs


# Example usage and registration
def create_custom_freeway_env(**kwargs):
    """Factory function to create custom freeway environment."""
    return CustomMinAtarFreeway(**kwargs)


# Example of how to register with pgx.make() (requires modifying core.py)
def register_custom_env():
    """
    To register this environment with pgx.make(), you would need to:
    1. Add "custom-freeway" to the EnvId literal in core.py
    2. Add a case in the make() function in core.py:
    
    elif env_id == "custom-freeway":
        from custom_freeway import CustomMinAtarFreeway
        return CustomMinAtarFreeway()
    """
    pass
