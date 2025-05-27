import torch as th
from torch import nn
from torch.distributions import Normal
from gymnasium import spaces

from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.type_aliases import PyTorchObs


# Create a custom distribution class with sigmoid instead of tanh
class SigmoidDiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a sigmoid transformation.
    """

    def __init__(self, action_dim: int, sigmoid_squash: float):
        super().__init__()
        self.action_dim = action_dim
        self.distribution = None
        self.gaussian_actions = None
        self.sigmoid_squash = sigmoid_squash

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> tuple[nn.Module, nn.Parameter]:
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "SigmoidDiagGaussianDistribution":
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # Inverse of parameterized sigmoid: sigmoid^-1(y) = log(y/(1-y)) * sigmoid_squash
        x = th.log(actions / (1 - actions + 1e-10) + 1e-10) * self.sigmoid_squash
        log_prob = self.distribution.log_prob(x)

        # Apply correction term for the sigmoid transformation
        # For parameterized sigmoid, we need to multiply by sigmoid_squash
        log_prob -= th.log(actions * (1 - actions) + 1e-10) + th.log(th.tensor(self.sigmoid_squash))

        # Sum along action dimension
        return log_prob.sum(dim=1)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy().sum(dim=1)

    def sample(self) -> th.Tensor:
        # Sample from Gaussian
        self.gaussian_actions = self.distribution.rsample()
        # Apply parameterized sigmoid to bound actions between 0 and 1
        return th.sigmoid(self.gaussian_actions / self.sigmoid_squash)

    def mode(self) -> th.Tensor:
        # Mode of the Gaussian is the mean
        self.gaussian_actions = self.distribution.mean
        # Apply parameterized sigmoid to bound actions between 0 and 1
        return th.sigmoid(self.gaussian_actions / self.sigmoid_squash)

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor,
                            deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(mean_actions, log_std)
        return self.mode() if deterministic else self.sample()

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


# Custom Actor class using sigmoid instead of tanh
class SigmoidActor(Actor):
    def __init__(self, *args, sigmoid_squash: float, **kwargs):
        # Extract sigmoid_squash before passing kwargs to parent
        self.sigmoid_squash = sigmoid_squash
        # Just pass through all arguments to the parent class
        super().__init__(*args, **kwargs)

        if self.use_sde:
            raise ValueError("SDE is not supported for sigmoid distribution")
        else:
            # Replace the action distribution with our sigmoid-based one
            action_dim = self.action_space.shape[0]
            self.action_dist = SigmoidDiagGaussianDistribution(
                action_dim, sigmoid_squash=self.sigmoid_squash)
            last_layer_dim = self.net_arch[-1] if len(self.net_arch) > 0 else self.features_dim
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Get actions bounded in [0, 1] through sigmoid
        actions = self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs)
        # Rescale to actual action space - convert NumPy arrays to PyTorch tensors
        low = th.as_tensor(self.action_space.low, device=actions.device, dtype=actions.dtype)
        high = th.as_tensor(self.action_space.high, device=actions.device, dtype=actions.dtype)
        return low + (high - low) * actions


# Custom SAC policy that uses the SigmoidActor
class SigmoidSACPolicy(SACPolicy):
    def __init__(self, *args, sigmoid_squash: float, **kwargs):
        self.sigmoid_squash = sigmoid_squash
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        # Add sigmoid_squash to actor_kwargs
        actor_kwargs["sigmoid_squash"] = self.sigmoid_squash
        return SigmoidActor(**actor_kwargs).to(self.device)
