import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import os 
import gym
import random
from replaybuffer import ReplayBuffer
import math


class SACActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),

            nn.Flatten(),

            nn.Linear(82944 , 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, obs):
        return torch.tanh(self.net(obs).chunk(2, dim=-1)[0])
    
    def sample(self, obs):
        means, log_stds =  self.net(obs).chunk(2, dim=-1)

        return self.reparameterize(means, log_stds)
    
    def calculate_log_pi(self, log_stds, noises, actions):
        """ 確率論的な行動の確率密度を返す． """
        # ガウス分布 `N(0, stds * I)` における `noises * stds` の確率密度の対数(= \log \pi(u|a))を計算する．
        log_pis = -0.5 * math.log(2 * math.pi) * log_stds.size(-1) - log_stds.sum(dim=-1, keepdim=True) - (0.5 * noises.pow(2)).sum(dim=-1, keepdim=True)
        log_pis -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return log_pis

    def reparameterize(self, means, log_stds):
        stds = log_stds.exp()
        noises = torch.randn_like(means)

        # Reparameterization Trick
        us = means + noises * stds

        # tanhを適用し，確率論的な行動を計算する．
        actions = torch.tanh(us)

        # 確率論的な行動の確率密度の対数を計算する．
        log_pis = self.calculate_log_pi(log_stds, noises, actions)

        return actions, log_pis

    


class SACCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(3, 32, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),

            nn.Flatten(),

            nn.Linear(64*48*48 , 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(3, 32, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),

            nn.Flatten(),

            nn.Linear(82944 , 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return self.net1(x), self.net2(x)
    


class SAC():
    def __init__(self, rb, device=torch.device('cuda'),
                 batch_size=256, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4,
                 replay_size=10**4, start_steps=3, tau=5e-3, alpha=0.2, reward_scale=1.0):
        super().__init__()

        self.actor = SACActor().to(device)
        self.critic = SACCritic().to(device)
        self.critic_target = SACCritic().to(device).eval()

          # ターゲットネットワークの重みを初期化し，勾配計算を無効にする．
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # オプティマイザ．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # その他パラメータ．
        self.learning_steps = 0
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.start_steps = start_steps
        self.tau = tau
        self.alpha = alpha
        self.reward_scale = reward_scale
        self.buffer= rb


    def is_update(self, steps):
        # 学習初期の一定期間(start_steps)は学習しない．
        return steps >= max(self.start_steps, self.batch_size)
    

    def update(self):
        self.learning_steps += 1

        actions, states, next_states, rewards, dones = self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()


    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)

            #clipped double Q
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis

        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)

        self.optim_critic.step()


    def update_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)

        loss_actor = (self.alpha * log_pis - torch.min(qs1, qs2)).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)

        self.optim_actor.step()


    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            #soft target update
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)


    def explore(self, state):
        """ 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す． """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()


    def exploit(self, state):
        """ 決定論的な行動を返す． """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]
    

    #add to buffer
    def add_experience(self, action, state, new_state,reward, done):
        self.buffer.add(action, state, new_state, reward, done)


    #save function
    def save(self, path='models/SAC_kuka'):
        if 'models' in path and os.path.isdir('models') is False:
            os.mkdir('models')

        torch.save({'actor_weights': self.actor.state_dict(),
                    'critic_weights': self.critic.state_dict(),
                    'Coptimizer_param': self.optim_critic.state_dict(),
                    'Aoptimizer_param': self.optim_actor.state_dict()
                    }, path)
        
        print("Saved Model Weights!")


    def load(self, path='models/SAC_kuka'):
        model_dict = torch.load(path)
        self.actor.load_state_dict(model_dict['actor_weights'])
        self.critic.load_state_dict(model_dict['critic_weights'])
        self.optim_critic.load_state_dict(model_dict['Coptimizer_param'])
        self.optim_actor.load_state_dict(model_dict['Aoptimizer_param'])

        print("Model Weights Loaded!")