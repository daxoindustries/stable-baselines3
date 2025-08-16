import torch as th
import numpy as np
from typing import Any, Optional, Union, ClassVar
from gymnasium import spaces

from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.sac.policies import DecentralizedSACPolicy
from stable_baselines3.common.policies import BasePolicy


class DecentralizedSAC(SAC):
    """
    Decentralized SAC for multi-agent scenarios where agents share the same
    policy.

    The observation format is expected to be: (num_agents, agent_obs_dim)
    The action format is expected to be: (num_agents, agent_action_dim)

    Each agent gets its individual observation as input. The policy is run
    multiple times (once per agent) and actions are aggregated into the
    structured format.

    Each environment step generates num_agents separate transitions in the
    replay buffer, so training works like regular SAC on individual agent data.

    :param policy: The policy model to use (must be "DecentralizedSACPolicy")
    :param env: The environment to learn from (must have structured obs/action spaces)
    :param num_agents: Number of agents sharing the same policy
    """

    # Override policy aliases to only include DecentralizedSACPolicy
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "DecentralizedSACPolicy": DecentralizedSACPolicy,
    }
    policy: DecentralizedSACPolicy  # Type annotation for the instance variable

    def __init__(
        self,
        policy: Union[str, type[BasePolicy]] = "DecentralizedSACPolicy",
        env: Union[GymEnv, str] = None,
        num_agents: int = None,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # Validate that only DecentralizedSACPolicy is used
        if isinstance(policy, str):
            if policy != "DecentralizedSACPolicy":
                raise ValueError("DecentralizedSAC only supports 'DecentralizedSACPolicy'")
        elif policy != DecentralizedSACPolicy:
            raise ValueError("DecentralizedSAC only supports DecentralizedSACPolicy")

        self.num_agents = num_agents

        # Inject num_agents into policy_kwargs (individual_obs will be calculated from env)
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs.update({
            "num_agents": num_agents,
        })

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        # Validate structured action space
        if len(self.env.action_space.shape) != 2:
            raise ValueError(
                f"Action space must have shape (num_agents, agent_action_dim), got {self.env.action_space.shape}")
        if self.env.action_space.shape[0] != self.num_agents:
            raise ValueError(
                f"Action space first dimension ({self.env.action_space.shape[0]}) must equal num_agents ({self.num_agents})")

        if len(self.env.observation_space.shape) != 2:
            raise ValueError(
                f"Observation space must have shape (num_agents, agent_obs_dim), got {self.env.observation_space.shape}")
        if self.env.observation_space.shape[0] != self.num_agents:
            raise ValueError(
                f"Observation space first dimension ({self.env.observation_space.shape[0]}) must equal num_agents ({self.num_agents})")

        action_dim_per_agent = self.env.action_space.shape[1]
        individual_obs_dim = self.env.observation_space.shape[1]

        # Do all standard SAC setup first
        super()._setup_model()

        # Extract individual agent observation space from structured format (num_agents, agent_obs_dim)
        individual_obs_space = spaces.Box(
            low=self.observation_space.low[0],   # Bounds for first agent (assuming homogeneous)
            high=self.observation_space.high[0],
            shape=(individual_obs_dim,),
            dtype=self.observation_space.dtype
        )

        # Extract individual agent action space from structured format (num_agents, agent_action_dim)
        individual_action_space = spaces.Box(
            low=self.action_space.low[0],   # Bounds for first agent (assuming homogeneous)
            high=self.action_space.high[0],
            shape=(action_dim_per_agent,),
            dtype=self.action_space.dtype
        )

        # Replace replay buffer with one that uses individual agent spaces
        # since we store individual agent transitions, not multi-agent transitions
        replay_buffer_kwargs = self.replay_buffer_kwargs.copy()

        # Handle HerReplayBuffer special case
        from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
        if issubclass(self.replay_buffer_class, HerReplayBuffer):
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
            replay_buffer_kwargs["env"] = self.env

        # Create new replay buffer with individual agent spaces
        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            individual_obs_space,      # Use individual agent obs space
            individual_action_space,   # Use individual agent action space
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **replay_buffer_kwargs,
        )

        # Replace policy with correct observation and action spaces
        self.policy = self.policy_class(
            individual_obs_space,    # Individual agent obs space
            individual_action_space,  # Individual agent action space
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Recreate aliases since we replaced the policy
        self._create_aliases()

        # Adjust target entropy for individual agents if auto
        if self.target_entropy == "auto":
            raise ValueError("target_entropy must be set manually for DecentralizedSAC")

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict],
        reward: np.ndarray,
        dones: np.ndarray,
        infos,
    ) -> None:
        """
        Store individual agent transitions instead of full multi-agent transition.
        Each environment step generates num_agents separate transitions.

        Store transition in the replay buffer the same way as OffPolicyAlgorithm._store_transition.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).
        """
        # Store only the unnormalized version (following SB3 pattern)
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference (following SB3 pattern)
        from copy import deepcopy
        next_obs = deepcopy(new_obs_)

        # Handle terminal observations (following OffPolicyAlgorithm pattern)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    # Dict observations not implemented for DecentralizedSAC yet
                    raise NotImplementedError(
                        "Dict observations with terminal_observation not yet supported")
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        obs = self._last_original_obs

        # Store one transition per agent
        for agent_idx in range(self.num_agents):
            # Extract agent observations - ensure consistent batch dimension
            if obs.ndim == 2:
                # Single env: obs is (num_agents, agent_obs_dim)
                agent_obs = obs[agent_idx].reshape(1, -1)
                agent_next_obs = next_obs[agent_idx].reshape(1, -1)
            else:
                # Multiple envs: obs is (num_envs, num_agents, agent_obs_dim)
                agent_obs = obs[:, agent_idx]
                agent_next_obs = next_obs[:, agent_idx]

            # Extract agent's action from structured action space
            if buffer_action.ndim == 2:
                # Single env: buffer_action is (num_agents, agent_action_dim)
                agent_action = buffer_action[agent_idx].reshape(1, -1)
            else:
                # Multiple envs: buffer_action is (num_envs, num_agents, agent_action_dim)
                agent_action = buffer_action[:, agent_idx]

            replay_buffer.add(
                agent_obs,
                agent_next_obs,
                agent_action,
                reward_,
                dones,
                infos,
            )

        # Update state (following SB3 pattern)
        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
