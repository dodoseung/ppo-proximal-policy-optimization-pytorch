# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import numpy as np
import gym
from collections import deque

class ReplayBuffer():
    def __init__(self):
        super(ReplayBuffer, self).__init__()
        self.memory = []
        
    # Add the replay memory
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Sample the replay memory
    def sample(self):
        batch = self.memory
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    # Reset the replay memory
    def reset(self):
        self.memory = []

class ContinousPolicyNet(nn.Module):
    def __init__(self, state_num, min_action, max_action):
        super(ContinousPolicyNet, self).__init__()
        self.min_action = min_action
        self.max_action = max_action
        
        self.input = nn.Linear(state_num, 128)
        self.mu = nn.Linear(128, 1)
        self.std = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        mu = (self.max_action - self.min_action) * F.sigmoid(self.mu(x)) + self.min_action
        std = (self.max_action - self.min_action) * F.sigmoid(self.std(x)) / 2
        # mu = self.mu(x)
        # mu = mu.clamp(min=self.min_action, max=self.max_action)
        # std = F.softplus(self.std(x)) # eliminate nagative value

        return mu, std

class CriticNet(nn.Module):
    def __init__(self, state_num):
        super(CriticNet, self).__init__()
        self.input = nn.Linear(state_num, 128)
        self.output = nn.Linear(128, 1)
    
    def forward(self, x):      
        x = F.relu(self.input(x))
        value = self.output(x)
        return value
    
class PPO():
    def __init__(self, env, gamma=0.99, learning_rate=1e-3, lambd=0.95, K=3, T=20, eps=0.1, av_norm=False):
        super(PPO, self).__init__()
        self.env = env
        self.state_num = self.env.observation_space.shape[0]
        self.action_min = float(env.action_space.low[0])
        self.action_max = float(env.action_space.high[0])
          
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy (actor)
        self.actor_net = ContinousPolicyNet(self.state_num, self.action_min, self.action_max).to(self.device)
        self.actor_opt = optim.Adam(self.actor_net.parameters(), lr=learning_rate)
        
        # Critic
        self.critic_net = CriticNet(self.state_num).to(self.device)
        self.critic_opt = optim.Adam(self.critic_net.parameters(), lr=learning_rate)
        
        # Rollout
        self.memory = ReplayBuffer()
        self.T = T
        
        # Learning setting
        self.gamma = gamma
        
        # Generalized advantage estimation
        self.lambd = lambd
        
        # Learning epoch per a minibatch
        self.K = K
        
        # advantage clipping parameter
        self.eps = eps
        
        # Advantage normalization
        self.av_norm = av_norm
        
    # Get the action
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        mu, std = self.actor_net(state)
        action = D.Normal(mu, std).sample()
        action = action.cpu().detach().numpy()
        return action[0]
    
    # Generalized advantage estimation
    def gae(self, states, rewards, next_states, dones):
        # Get values
        values = self.critic_net(states)
        next_values = self.critic_net(next_states)

        # Get delta values
        target_td = rewards + self.gamma * next_values * (1-dones)
        delta = target_td - values
        delta = delta.detach().numpy()

        # Get advantages
        advantages = []
        advantage = 0
        
        for idx, d in enumerate(reversed(delta)):
            advantage = self.gamma * self.lambd * advantage + d if dones[len(dones)-idx-1] == 0 else d
            advantages.append(advantage)
        advantages = advantages[::-1]
  
        return advantages, target_td
    
    def learn(self):
        # Get memory from rollout
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)

        # Get pi theta old
        mu_old, std_old = self.actor_net(states)
        dist_old = D.Normal(mu_old, std_old)
        log_probs_old = dist_old.log_prob(actions)
        log_probs_old = log_probs_old.detach()

        # Get a, v, target from generalized advantage estimation
        advantages, target_td = self.gae(states, rewards, next_states, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = advantages.detach()

        # Normalize advantages
        advantages = ((advantages - advantages.mean()) / advantages.std()) if len(advantages) > 1 and self.av_norm else advantages
        
        # K epoch per a minibatch
        for _ in range(self.K):
            # Get pi theta new
            mu_new, std_new = self.actor_net(states)
            dist_new = D.Normal(mu_new, std_new)
            log_probs_new = dist_new.log_prob(actions)

            # Calculate ratio values
            ratio = torch.exp(log_probs_new - log_probs_old) # new/old, more stable
            
            # Calculate surrogate objectives and actor loss
            surrograte_1 = ratio * advantages
            surrograte_2 = ratio.clamp(min=1-self.eps, max=1+self.eps) * advantages
            
            # policy_loss_1 = surrograte_1[surrograte_1.abs() < surrograte_2.abs()]
            # policy_loss_2 = surrograte_2[surrograte_1.abs() >= surrograte_2.abs()]
            # policy_loss = torch.cat([policy_loss_1, policy_loss_2], dim=-1)
            
            # # Calculate an entropy loss
            # entropy_loss = 0.01 * dist_new.entropy()
            
            # # Calculate the critic loss and optimize the critic network
            # actor_loss = - (policy_loss.mean() + entropy_loss.mean())
            
            actor_loss = - torch.min(surrograte_1, surrograte_2).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), 1)
            self.actor_opt.step()
            
            # Calculate the critic loss and optimize the critic network
            critic_loss = F.smooth_l1_loss(self.critic_net(states), target_td.detach())
            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1)
            self.critic_opt.step()
            
        # Reset the memory
        self.memory.reset()
            
        
def main():
    env = gym.make("Pendulum-v0")
    agent = PPO(env, gamma=0.9, learning_rate=2e-4, lambd=0.9, K=100, T=2000, eps=0.2, av_norm=False)
    ep_rewards = deque(maxlen=20)
    total_episode = 10000
    
    for i in range(total_episode):
        state = env.reset()
        rewards = []

        while True:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.memory.add(state, action, reward / 10, next_state, done)             
            rewards.append(reward)

            # PPO
            if len(agent.memory.memory) == agent.T:
                agent.learn()
            
            if done:
                ep_rewards.append(sum(rewards))
                
                if i % 20 == 0:
                    print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                break
            
            

            state = next_state
    

if __name__ == '__main__':
    main()
    