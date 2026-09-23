import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class ActorCriticMethod(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, aux_lambda=0.1):
        super().__init__()
        self.aux_lambda = aux_lambda

        # 1. Shared feature trunk
        self.shared_trunk = nn.sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        # 2. Actor Head
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # 3. Critic Head
        self.critic = nn.Linear(64, 1)

        # 4. Dynamics head
        self.dynamics_head = nn.Sequential(
            nn.Linear(64 + action_dim, 32),
            nn.Tanh(),
            nn.Linear(32, state_dim)
        )
    
    def get_features(self, state):
        """ The latent embeddings z_t """
        return self.shared_trunk(state)

    def get_action_and_value(self, state, action=None):
        """ Used during both rollout collection(sample collections at random actions) and PPO update steps. """
        features = self.get_features(state)

        # Compute action distribution
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        action_log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(features).squeeze(-1)

        return action, action_log_prob, entropy, value, features

    def predict_next_state(self, features, action, detach_trunk=False):
        """
        Predicts s_{t+1} from latent z_t and executed action a_t.
        If detach_trunk=Ture or setting lambda = 0(used for baseline mode), gradients do not flow into trunk which allows the dynamics head to learn to predict next state but gradients are stopped before they reach the shared trunk.
        """
        if detach_trunk:
            # baseline mode
            features = features.detach()

        dynamics_input = torch.cat([features, action], dim=-1)
        return self.dynamics_head(dynamics_input)