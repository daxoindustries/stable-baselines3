import torch as th
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import StateDependentNoiseDistribution, SquashedDiagGaussianDistribution
from gymnasium import spaces


class LatentActor(Actor):
    """
    Actor network with a latent policy + decoder architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Features extractor to use
    :param features_dim: Dimension of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_actions,)
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation
    :param clip_mean: Clip the mean output to avoid numerical instability
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param decoder_net_arch: Architecture of the decoder network (can be empty list [] for direct mapping)
    :param freeze_latent_policy: Whether to freeze the latent policy or not
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: Union[List[int], Dict[str, List[int]]],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = True,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        # Latent policy parameters
        decoder_net_arch: List[int] = None,
        freeze_latent_policy: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )
        if decoder_net_arch is None:
            raise ValueError(
                "decoder_net_arch must be provided (can be an empty list [] for direct mapping)")

        # Store parameters as attributes
        self.decoder_net_arch = decoder_net_arch
        self.freeze_latent_policy = freeze_latent_policy

        # Feature net that captures high level action plans
        latent_features = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_features = nn.Sequential(*latent_features)

        # Get the output dimension from the last layer of net_arch
        latent_feature_output_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        # Create decoder network
        latent_decoder = create_mlp(latent_feature_output_dim, -1, decoder_net_arch, activation_fn)
        self.latent_decoder = nn.Sequential(*latent_decoder)

        # Override latent_pi to create the latent policy network
        self.latent_pi = nn.Sequential(self.latent_features, self.latent_decoder)

        # Create mu, log_std and sde heads
        last_layer_dim = decoder_net_arch[-1] if len(
            decoder_net_arch) > 0 else latent_feature_output_dim
        action_dim = get_action_dim(self.action_space)
        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std, use_expln, learn_features=True, squash_output=True)
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

        # Whether to freeze the latent policy
        if freeze_latent_policy:
            for param in self.latent_features.parameters():
                param.requires_grad = False

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                decoder_net_arch=self.decoder_net_arch,
                freeze_latent_policy=self.freeze_latent_policy,
            )
        )
        return data

    def load_latent_policy(self, path: str) -> None:
        """
        Load the latent_features from a full model zip file.

        :param path: Path to the saved model (.zip file)
        """
        if not path.endswith('.zip'):
            raise ValueError(f"Only .zip files are supported. Got: {path}")

        # Load latent_features from the full model zip file
        state_dict = self._extract_latent_features_from_zip(path)
        self.latent_features.load_state_dict(state_dict)

        if self.freeze_latent_policy:
            for param in self.latent_features.parameters():
                param.requires_grad = False

    def _extract_latent_features_from_zip(self, zip_path: str) -> Dict[str, th.Tensor]:
        """
        Extract latent_features state_dict from a full model zip file.

        :param zip_path: Path to the saved model (.zip file)
        :return: State dict for latent_features only
        """
        from stable_baselines3.common.save_util import load_from_zip_file

        # Load only the parameters from the zip file
        _, params, _ = load_from_zip_file(zip_path, device='cpu', load_data=False)

        if 'policy' not in params:
            raise ValueError(
                f"No 'policy' found in {zip_path}. Make sure this is a valid Latent SAC model.")

        policy_state_dict = params['policy']

        # Extract latent_features parameters
        latent_features_state_dict = {}
        prefix = 'actor.latent_features.'

        for key, value in policy_state_dict.items():
            if key.startswith(prefix):
                # Remove the prefix to get the relative key for latent_features
                relative_key = key[len(prefix):]
                latent_features_state_dict[relative_key] = value

        if not latent_features_state_dict:
            raise ValueError(
                f"No latent_features found in {zip_path}. Make sure this is a LatentSAC model.")

        return latent_features_state_dict


class LatentSACPolicy(SACPolicy):
    """
    Policy class with latent policy + decoder architecture for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: Network architecture
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation
    :param clip_mean: Clip the mean output to avoid numerical instability
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic
    :param decoder_net_arch: Architecture of the decoder network (can be empty list [] for direct mapping)
    :param freeze_latent_policy: Whether to freeze the latent policy or not
    :param pretrained_latent_policy_path: Path to a pretrained latent policy
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = True,  # Default to True for decoder SDE
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Optional[Type[nn.Module]] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        # Latent policy parameters
        decoder_net_arch: List[int] = None,
        freeze_latent_policy: bool = False,
        pretrained_latent_policy_path: Optional[str] = None,
    ):
        self.decoder_net_arch = decoder_net_arch
        self.freeze_latent_policy = freeze_latent_policy
        self.pretrained_latent_policy_path = pretrained_latent_policy_path

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[nn.Module] = None) -> LatentActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)

        # Add latent-specific parameters
        actor_kwargs.update({
            "decoder_net_arch": self.decoder_net_arch,
            "freeze_latent_policy": self.freeze_latent_policy,
        })

        actor = LatentActor(**actor_kwargs).to(self.device)

        # Load pretrained latent policy if provided
        if self.pretrained_latent_policy_path is not None:
            actor.load_latent_policy(self.pretrained_latent_policy_path)

        return actor

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                decoder_net_arch=self.decoder_net_arch,
                freeze_latent_policy=self.freeze_latent_policy,
                pretrained_latent_policy_path=self.pretrained_latent_policy_path,
            )
        )
        return data
