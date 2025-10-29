"""Example usage of custom PGX environment

This demonstrates how to use the custom MinAtarFreeway environment
and how to integrate it with existing code.
"""
import jax
import jax.numpy as jnp
from custom_freeway import CustomMinAtarFreeway, create_custom_freeway_env


def example_direct_usage():
    """Example 1: Direct instantiation and usage"""
    print("=== Example 1: Direct Usage ===")
    
    # Create custom environment with modified parameters
    env = CustomMinAtarFreeway(
        use_minimal_action_set=True,
        sticky_action_prob=0.1,
        reward_scale=2.0,  # Double rewards
        player_speed=2,    # Slower than original (3)
        time_limit=3000,   # Longer than original (2500)
    )
    
    # Initialize environment
    key = jax.random.PRNGKey(42)
    state = env.init(key)
    
    print(f"Environment ID: {env.id}")
    print(f"Version: {env.version}")
    print(f"Number of players: {env.num_players}")
    print(f"Observation shape: {env.observation_shape}")
    print(f"Number of actions: {env.num_actions}")
    print(f"Initial position: {state._pos}")
    print(f"Initial total rewards: {state._total_rewards}")
    print(f"Initial crossings: {state._crossings}")
    
    # Run a few steps
    for step in range(5):
        key, action_key = jax.random.split(key)
        action = jax.random.randint(action_key, (), 0, env.num_actions)
        
        state = env.step(state, action, key)
        
        print(f"Step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Position: {state._pos}")
        print(f"  Reward: {state.rewards[0]}")
        print(f"  Total rewards: {state._total_rewards}")
        print(f"  Crossings: {state._crossings}")
        print(f"  Terminated: {state.terminated}")
        print()


def example_factory_function():
    """Example 2: Using factory function"""
    print("=== Example 2: Factory Function ===")
    
    # Create environment using factory function
    env = create_custom_freeway_env(
        reward_scale=1.5,
        player_speed=4,  # Faster than default
        time_limit=2000,  # Shorter than default
    )
    
    key = jax.random.PRNGKey(123)
    state = env.init(key)
    
    print(f"Custom environment created with ID: {env.id}")
    print(f"Player speed: {env.player_speed}")
    print(f"Time limit: {env.time_limit}")
    print(f"Reward scale: {env.reward_scale}")


def example_jit_compilation():
    """Example 3: JIT compilation for performance"""
    print("=== Example 3: JIT Compilation ===")
    
    env = CustomMinAtarFreeway()
    
    # JIT compile the step function for better performance
    @jax.jit
    def jit_step(state, action, key):
        return env.step(state, action, key)
    
    @jax.jit
    def jit_init(key):
        return env.init(key)
    
    # Use JIT compiled functions
    key = jax.random.PRNGKey(456)
    state = jit_init(key)
    
    print("JIT compiled environment functions")
    print(f"Initial state created with JIT: {state._pos}")
    
    # Run steps with JIT
    for i in range(3):
        key, action_key = jax.random.split(key)
        action = jax.random.randint(action_key, (), 0, env.num_actions)
        state = jit_step(state, action, key)
        print(f"JIT step {i + 1}: pos={state._pos}, reward={state.rewards[0]}")


def example_vectorized_environment():
    """Example 4: Vectorized environment for parallel execution"""
    print("=== Example 4: Vectorized Environment ===")
    
    env = CustomMinAtarFreeway()
    
    # Create multiple environments in parallel
    num_envs = 4
    keys = jax.random.split(jax.random.PRNGKey(789), num_envs)
    
    # Initialize multiple environments
    states = jax.vmap(env.init)(keys)
    
    print(f"Created {num_envs} parallel environments")
    print(f"Positions: {states._pos}")
    print(f"Total rewards: {states._total_rewards}")
    
    # Run one step in all environments
    action_keys = jax.random.split(jax.random.PRNGKey(999), num_envs)
    actions = jax.random.randint(action_keys, (num_envs,), 0, env.num_actions)
    
    # Vectorized step
    next_states = jax.vmap(env.step)(states, actions, action_keys)
    
    print(f"After one step:")
    print(f"Positions: {next_states._pos}")
    print(f"Rewards: {next_states.rewards.flatten()}")
    print(f"Total rewards: {next_states._total_rewards}")


def example_integration_with_existing_code():
    """Example 5: Integration with existing training code"""
    print("=== Example 5: Integration Example ===")
    
    # This shows how you might integrate with existing code
    # that expects a pgx environment
    
    def train_agent(env, num_steps=100):
        """Example training function that works with any pgx environment"""
        key = jax.random.PRNGKey(0)
        state = env.init(key)
        
        total_reward = 0.0
        
        for step in range(num_steps):
            key, action_key = jax.random.split(key)
            action = jax.random.randint(action_key, (), 0, env.num_actions)
            
            state = env.step(state, action, key)
            total_reward += state.rewards[0]
            
            if state.terminated:
                print(f"Episode ended at step {step + 1}")
                break
        
        return total_reward
    
    # Use with custom environment
    custom_env = CustomMinAtarFreeway(reward_scale=3.0)
    reward = train_agent(custom_env, 50)
    print(f"Custom environment total reward: {reward}")
    
    # Use with original environment (if available)
    try:
        import pgx
        original_env = pgx.make("minatar-freeway")
        original_reward = train_agent(original_env, 50)
        print(f"Original environment total reward: {original_reward}")
    except ImportError:
        print("PGX not available for comparison")


if __name__ == "__main__":
    print("Custom PGX Environment Usage Examples")
    print("=" * 50)
    
    example_direct_usage()
    example_factory_function()
    example_jit_compilation()
    example_vectorized_environment()
    example_integration_with_existing_code()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
