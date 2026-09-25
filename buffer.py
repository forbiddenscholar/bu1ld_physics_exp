import numpy as np
import torch

class RolloutBuffer: 
    def __init__(self, buffer_size, state_dim, action_dim, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre allocate memory buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32) # indicate episode termination (1 if terminate, else 0, used for resettting at boundary episode)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)

        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0

    def add(self, state, action, reward, done, value, log_prob, next_state):
        """ Append a new trajectory step into the buffer"""

        assert self.ptr < self.buffer_size, "Buffer overflow"

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.next_states[self.ptr] = next_state

        self.ptr += 1

# return is the G_t here : G_t = r_t + gamma*r_t+1 + gamma^2*r_t+2 .....
# This is served as a ground truth because critic is a neural network and requires a regression label while training
# Advantage = G_t - V(s_t) (Actual return - critic value)
# G_t = Advantage + V(s_t) 
# self.returns = self.advantages + self.values
# Loss_critic = Mean((V(s_t) - returns_t)**2)

    def compute_gae(self, last_value, last_done):
        """
        Computes Generalized Advantage Estimation (GAE) backwards in time.
        last_value: Critic V(s) estimate for the state immediately after the final step.
        last_done:  Whether that state was a terminal boundary.
        """
        last_gae = 0.0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
                
            # TD Error: delta = r + gamma * V(s') - V(s)
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            
            # Recursive GAE: A_t = delta + (gamma * lambda) * A_{t+1}
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            
        # Target returns for Critic regression: G_t = A_t + V(s_t)
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """Yields randomized mini-batches converted to PyTorch tensors."""
        indices = np.random.permutation(self.buffer_size)
        
        # Convert arrays to tensors
        b_states = torch.as_tensor(self.states, dtype=torch.float32)
        b_actions = torch.as_tensor(self.actions, dtype=torch.float32)
        b_log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32)
        b_advantages = torch.as_tensor(self.advantages, dtype=torch.float32)
        b_returns = torch.as_tensor(self.returns, dtype=torch.float32)
        b_next_states = torch.as_tensor(self.next_states, dtype=torch.float32)

        # Standardize advantages across the entire rollout batch (stabilizes gradient updates)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        for start_idx in range(0, self.buffer_size, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield (
                b_states[batch_idx],
                b_actions[batch_idx],
                b_log_probs[batch_idx],
                b_advantages[batch_idx],
                b_returns[batch_idx],
                b_next_states[batch_idx],
            )

    def reset(self):
        self.ptr = 0