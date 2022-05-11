# Code 2.1 from Foundations of Deep Reinforcement Learning by Laura Graesser and Wah Loon Keng
# A standalone working implementation of REINFORCE for CartPole-v0

from torch.distributions import Categorical
import gym
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

gamma = 0.99
 
class Pi(nn.Module):
    # constructs the policy network
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train() # set tge training mode

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    # defines the method to produce action
    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32)) # to sensor
        pdparam = self.forward(x) # forward pass
        pd = Categorical(logits = pdparam) # probability distribution
        action = pd.sample() # pi (a | s ) in action via pd
        log_prob = pd.log_prob(action) # log_prob of pi(a|s)
        self.log_probs.append(log_prob) # store for training
        return action.item()

def train(pi, optimizer):
    # inner gradient-ascent loop of REINFORCE algorithm 
    T = len(pi.rewards)
    rets = np.empty(T, dtype = np.float32)
    future_ret = 0.0
    # compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets # gradient term; negative for maximizing
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward() # backpropagate, compute gradients
    optimizer.step() # gradient-ascent, update the weights
    return loss
    
def main():
    env = gym.make('CartPole-v0')
    in_dim = env.observation_space.shape[0] # 4
    out_dim = env.action_space.n # 2
    pi = Pi(in_dim, out_dim) # policy pi_theta for REINFORCE
    optimizer = optim.Adam(pi.parameters(), lr = 0.01)
    for epi in range (300): 
        state = env.reset()
        for t in range(200): # cartpole max timestep is 200
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                break
        loss = train(pi, optimizer) # train per episode
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset() # onpolicy: clear memory after training
        print(f'Episode {epi}, loss: {loss}, \
        total_reward: {total_reward}, solved: {solved}')

if __name__ == '__main__':
    main()
