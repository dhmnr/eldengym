"""
Environment registration system for EldenGym.

Similar to Gymnasium's registration system, allows creating environments
by name using eldengym.make('env-id').
"""

from typing import Dict, Any, Callable, Optional
import copy


class EnvSpec:
    """Specification for a registered environment."""

    def __init__(
        self,
        id: str,
        entry_point: Callable,
        kwargs: Optional[Dict[str, Any]] = None,
        max_episode_steps: Optional[int] = None,
        setup_hook: Optional[Callable] = None,
    ):
        self.id = id
        self.entry_point = entry_point
        self.kwargs = kwargs or {}
        self.max_episode_steps = max_episode_steps
        self.setup_hook = setup_hook

    def make(self, **kwargs) -> Any:
        """Create an instance of the environment."""
        # Merge default kwargs with user-provided kwargs
        _kwargs = copy.deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        # Add max_episode_steps if specified
        if self.max_episode_steps is not None and "max_step" not in _kwargs:
            _kwargs["max_step"] = self.max_episode_steps

        # Create environment
        env = self.entry_point(**_kwargs)

        # Run setup hook if provided
        if self.setup_hook is not None:
            self.setup_hook(env)

        return env


class EnvRegistry:
    """Registry for environments."""

    def __init__(self):
        self.env_specs: Dict[str, EnvSpec] = {}

    def register(
        self,
        id: str,
        entry_point: Callable,
        kwargs: Optional[Dict[str, Any]] = None,
        max_episode_steps: Optional[int] = None,
        setup_hook: Optional[Callable] = None,
        force: bool = False,
    ):
        """
        Register an environment.

        Args:
            id: Environment ID (e.g., 'EldenRing-Margit-v0')
            entry_point: Callable that creates the environment
            kwargs: Default keyword arguments for the environment
            max_episode_steps: Maximum episode steps (sets max_step parameter)
            setup_hook: Optional function to call after environment creation
            force: If True, overwrite existing registration
        """
        if id in self.env_specs and not force:
            raise ValueError(
                f"Environment {id} already registered. Use force=True to overwrite."
            )

        spec = EnvSpec(
            id=id,
            entry_point=entry_point,
            kwargs=kwargs,
            max_episode_steps=max_episode_steps,
            setup_hook=setup_hook,
        )
        self.env_specs[id] = spec

    def make(self, id: str, **kwargs) -> Any:
        """
        Create an environment by ID.

        Args:
            id: Environment ID
            **kwargs: Additional keyword arguments to override defaults

        Returns:
            Environment instance
        """
        if id not in self.env_specs:
            raise ValueError(
                f"Environment {id} not found. Available environments: {list(self.env_specs.keys())}"
            )

        spec = self.env_specs[id]
        return spec.make(**kwargs)

    def list(self):
        """List all registered environments."""
        return list(self.env_specs.keys())


# Global registry instance
registry = EnvRegistry()


def register(
    id: str,
    entry_point: Callable,
    kwargs: Optional[Dict[str, Any]] = None,
    max_episode_steps: Optional[int] = None,
    setup_hook: Optional[Callable] = None,
    force: bool = False,
):
    """
    Register an environment in the global registry.

    Args:
        id: Environment ID (e.g., 'EldenRing-Margit-v0')
        entry_point: Callable that creates the environment
        kwargs: Default keyword arguments for the environment
        max_episode_steps: Maximum episode steps (sets max_step parameter)
        setup_hook: Optional function to call after environment creation.
                   Receives the environment instance as an argument.
        force: If True, overwrite existing registration

    Example:
        >>> def my_setup(env):
        ...     print(f"Setting up {env.scenario_name}")
        ...     env.custom_attr = "value"
        >>>
        >>> eldengym.register(
        ...     id='Custom-v0',
        ...     entry_point=EldenGymEnv,
        ...     kwargs={'scenario_name': 'margit'},
        ...     setup_hook=my_setup
        ... )
    """
    registry.register(id, entry_point, kwargs, max_episode_steps, setup_hook, force)


def make(id: str, **kwargs) -> Any:
    """
    Create an environment by ID.

    Args:
        id: Environment ID
        **kwargs: Additional keyword arguments to override defaults

    Returns:
        Environment instance

    Example:
        >>> import eldengym
        >>> env = eldengym.make('EldenRing-Margit-v0')
        >>> env = eldengym.make('EldenRing-Godrick-v0', game_speed=0.5)
    """
    return registry.make(id, **kwargs)


def list_envs():
    """List all registered environments."""
    return registry.list()
