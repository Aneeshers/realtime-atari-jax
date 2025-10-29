# Custom PGX Environment Creation Guide

This guide shows you how to create custom PGX environments by modifying existing ones, using the MinAtarFreeway environment as an example.

## Overview

PGX environments consist of two main components:

1. **State Class**: Inherits from `core.State` and contains all game state data
2. **Environment Class**: Inherits from `core.Env` and implements the game logic

## Key Components

### 1. State Class Structure

```python
@dataclass
class CustomState(core.State):
    # Required core fields
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((height, width, channels), dtype=jnp.bool_)
    rewards: Array = jnp.zeros(num_players, dtype=jnp.float32)
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(num_actions, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    
    # Game-specific fields
    _custom_field1: Array = jnp.array(default_value, dtype=jnp.int32)
    _custom_field2: Array = jnp.array(default_value, dtype=jnp.float32)
    
    @property
    def env_id(self) -> core.EnvId:
        return "your-custom-env-id"
```

### 2. Environment Class Structure

```python
class CustomEnv(core.Env):
    def __init__(self, **custom_params):
        super().__init__()
        # Store custom parameters
        self.custom_param = custom_params.get('custom_param', default_value)
        
    def _init(self, key: PRNGKey) -> CustomState:
        # Initialize game state
        pass
        
    def _step(self, state: core.State, action, key) -> CustomState:
        # Implement game step logic
        pass
        
    def _observe(self, state: core.State, player_id: Array) -> Array:
        # Generate observation from state
        pass
        
    @property
    def id(self) -> core.EnvId:
        return "your-custom-env-id"
        
    @property
    def version(self) -> str:
        return "v1"
        
    @property
    def num_players(self):
        return 1  # or number of players
```

## Step-by-Step Process

### Step 1: Copy and Modify the Original

1. Copy the original environment file (e.g., `pgx/minatar/freeway.py`)
2. Rename classes and functions to avoid conflicts
3. Modify parameters and behavior as needed

### Step 2: Update State Class

- Change the class name (e.g., `CustomFreewayState`)
- Add any new state fields you need
- Update the `env_id` property
- Keep all required `core.State` fields

### Step 3: Update Environment Class

- Change the class name (e.g., `CustomMinAtarFreeway`)
- Modify `__init__` to accept your custom parameters
- Update `_init`, `_step`, and `_observe` methods as needed
- Update the `id` property

### Step 4: Update Game Logic

- Modify the step function to implement your changes
- Update reward calculations
- Adjust game mechanics as desired
- Ensure all JAX operations are differentiable

### Step 5: Test Your Environment

```python
# Basic usage test
env = CustomMinAtarFreeway()
key = jax.random.PRNGKey(42)
state = env.init(key)

# Run a few steps
for i in range(10):
    key, action_key = jax.random.split(key)
    action = jax.random.randint(action_key, (), 0, env.num_actions)
    state = env.step(state, action, key)
    print(f"Step {i}: reward={state.rewards[0]}")
```

## Integration Options

### Option 1: Direct Import (Recommended)

```python
from custom_freeway import CustomMinAtarFreeway

env = CustomMinAtarFreeway(reward_scale=2.0)
state = env.init(jax.random.PRNGKey(0))
```

### Option 2: Factory Function

```python
from custom_freeway import create_custom_freeway_env

env = create_custom_freeway_env(reward_scale=2.0)
```

### Option 3: Register with pgx.make() (Advanced)

To register your environment with `pgx.make()`, you need to modify `pgx/core.py`:

1. Add your environment ID to the `EnvId` literal:
```python
EnvId = Literal[
    # ... existing environments ...
    "custom-freeway",
]
```

2. Add a case in the `make()` function:
```python
elif env_id == "custom-freeway":
    from custom_freeway import CustomMinAtarFreeway
    return CustomMinAtarFreeway()
```

## Best Practices

### 1. Maintain JAX Compatibility

- Use JAX operations (`jax.lax.cond`, `jax.lax.scan`, etc.)
- Ensure all functions are differentiable
- Use `jnp` instead of `np` for arrays

### 2. Preserve Core Interface

- Keep the same method signatures as the base class
- Maintain compatibility with `env.init()` and `env.step()`
- Ensure `legal_action_mask` is properly updated

### 3. Add Custom Parameters

- Use constructor parameters for customization
- Provide sensible defaults
- Document parameter effects

### 4. Test Thoroughly

- Test with different random seeds
- Verify JIT compilation works
- Test vectorized environments
- Compare behavior with original

### 5. Performance Considerations

- Use JIT compilation for training loops
- Consider vectorized environments for parallel execution
- Profile performance-critical sections

## Example Modifications

### Common Modifications

1. **Reward Scaling**: Multiply rewards by a factor
2. **Speed Changes**: Modify player or enemy movement speeds
3. **Time Limits**: Change episode length
4. **Difficulty**: Adjust spawn rates or enemy behavior
5. **Observation Space**: Add new observation channels

### Advanced Modifications

1. **New Game Mechanics**: Add power-ups, obstacles, etc.
2. **Multi-objective**: Add multiple reward signals
3. **Curriculum Learning**: Gradually increase difficulty
4. **Procedural Generation**: Randomize level layouts

## Troubleshooting

### Common Issues

1. **JAX Compilation Errors**: Ensure all operations are JAX-compatible
2. **Shape Mismatches**: Check array dimensions in state updates
3. **Gradient Issues**: Use `jax.lax.stop_gradient` where needed
4. **Performance**: Profile and optimize hot paths

### Debugging Tips

1. Use `jax.debug.print()` for debugging
2. Test with small examples first
3. Compare step-by-step with original implementation
4. Use JAX's `checkify` for runtime checks

## Files Created

- `custom_freeway.py`: Custom environment implementation
- `example_usage.py`: Usage examples and integration patterns
- `CUSTOM_PGX_GUIDE.md`: This comprehensive guide

## Next Steps

1. Run the example code to see the custom environment in action
2. Modify parameters to see how they affect behavior
3. Add your own custom modifications
4. Integrate with your training pipeline

The custom environment maintains full compatibility with PGX's interface while allowing you to experiment with different game mechanics and parameters.
