# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Main functions used in experiments

# +
import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer
# %matplotlib inline
import matplotlib.pyplot as plt
import sys

import random
import time
assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"
# -

# ## Environment: Windy gridworld
# Gives a reward of -1 for each step taken, while the final state is not reached

# +
from windy_gridworld import WindyGridworldEnv
env = WindyGridworldEnv()
# env??
# -

# ## Policy
#
# ### Target policy (choose greedy vs non-greedy)
# Greedy policy 

class GreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q):
        self.Q = Q
    
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """   
        
        # Inefficient but kept same structure as below if we change policy later
        probs = [1/(sum(self.Q[s] == max(self.Q[s]))) if Q[s][a] == np.max(self.Q[s]) else 0 for s,a in zip(states, actions)]

        return probs
        
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        # find out what the max action is
        best_action = np.where(self.Q[obs] == np.max(self.Q[obs]))[0]
        
        return best_action


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
        
    def get_probs(self, states, actions):
        # loop over the state action lists and compute probabilities according to eps greedy
#         probs = [(1-self.epsilon)/sum(self.Q[s] == max(self.Q[s])) if Q[s][a] == np.max(self.Q[s]) else self.epsilon/(self.Q.shape[1]-sum(self.Q[s] == max(self.Q[s]))) for s, a in zip(states, actions)]
        probs = [1 - self.epsilon if a == np.argmax(self.Q[s]) else epsilon/(self.Q.shape[1]-1) for s,a in zip(states, actions)]
        return probs
        
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
         
        actions = np.where(self.Q[obs] == np.max(self.Q[obs]))[0]
        p = np.random.uniform()
        if p > self.epsilon:
            # choose one of the best actions
            action = np.random.choice(actions)
        else:
            # return a random action
            action = np.random.randint(0,4)
                
        return action


# ### Behavioural policy
# Random policy from blackjack lab. 
# TODO: experiment with behavioural policies to check which yield interesting results

class RandomPolicy(object):
    """
    A behavioural policy
    """
    def __init__(self, nS, nA):
        self.probs = np.ones((nS, nA)) * 1/nA
        
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """        
        probs = [self.probs[s,a] for s,a in zip(states, actions)]
        
        return probs

    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        p_s = self.probs[state]
        
        return np.random.choice(range(0,self.probs.shape[1]), p=p_s)

random_policy = RandomPolicy(env.nS, env.nA)


# ## Monte Carlo

# ## Sampling function given an env and policy
# Function to sample an episode from the env.

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        state after the termination is not included in the list of states.
    """
    # initialize
    states = []
    actions = []
    rewards = []
    dones = []
    
    # get a starting state
    s = env.reset()
    d = False
    
    # keep looping until done, don's save the terminal state
    while not d:
        states.append(s)
        a = policy.sample_action(s)
        s, r, d, _ = env.step(a)
        
        # save                
        actions.append(a)
        rewards.append(r)
        dones.append(d)
        

    return states, actions, rewards, dones

# check the length of episodes that are generated for random policy
for episode in range(10):
    trajectory_data = sample_episode(env, random_policy)
#     print("Episode {}:\nStates {}\nActions {}\nRewards {}\nDones {}\n".format(episode,*trajectory_data))
    print(f"length of episode {episode}: {len(trajectory_data[0])}")


# +
# check the length of episodes that are generated for eps greedy policy
Q = np.zeros((env.nS, env.nA))
bp = EpsilonGreedyPolicy(Q, epsilon=0.1)

for episode in range(10):
    trajectory_data = sample_episode(env, bp)
#     print("Episode {}:\nStates {}\nActions {}\nRewards {}\nDones {}\n".format(episode,*trajectory_data))
    print(f"length of episode {episode}: {len(trajectory_data[0])}")


# -

# ## MC Ordinary Importance Sampling (make it work for windy gridworld)
# Status: updated to update Q instead of V

def Qdefaultdict2array(Q, nA, nS):
    Q_np = np.zeros((nS, nA))
    for S in range(nS):
        for A in range(nA):
            Q_np[S][A] = Q[S][A]
    return Q_np
            


def mc_ordinary_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode, epsilon=0.05):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    Q = defaultdict(lambda: defaultdict(float))
    returns_count = defaultdict(lambda: defaultdict(float))
    episode_lens = []
    
    # sample episodes
    for i in tqdm(range(num_episodes), position=0):
        # update behavioral function:
        behavior_policy = EpsilonGreedyPolicy(Qdefaultdict2array(Q, env.nA, env.nS), epsilon)
        # sample episode with new behavioural function
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        # save the episode length
        episode_lens.append(len(states))

        G = 0
        ratio = 1
        
        # loop backwards over the trajectory
        for timestep in range(len(states)-1, -1, -1):
            s = states[timestep]
            r = rewards[timestep]
            a = actions[timestep]
            G = discount_factor * G + r
            
            returns_count[s][a] += 1
            
            target_prob = target_policy.get_probs([s], [a])
            behavioral_prob = behavior_policy.get_probs([s], [a])  

            # update the weight
            ratio *= (target_prob[0])/(behavioral_prob[0])

            # use every visit incremental method
            Q[s][a] += 1/returns_count[s][a] * (ratio * G - Q[s][a])
                    
    Q = Qdefaultdict2array(Q, env.nA, env.nS)
    
    return Q, episode_lens


# ### MC: Weighted Importance Sampling
#
# ##### (TODO: Eventually: merge the two functions into one with a weighted flag)

def mc_weighted_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode, epsilon=0.05):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and weighted importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # create a matrix defaultdict for the Q function and the sum of weights C
    Q = defaultdict(lambda: defaultdict(float))
    C = defaultdict(lambda: defaultdict(float))
    episode_lens = []
    
    # sample episodes
    for i in tqdm(range(num_episodes), position=0):
        # update behavioral function:
        behavior_policy = EpsilonGreedyPolicy(Qdefaultdict2array(Q, env.nA, env.nS), epsilon)
        
        # sample episode with new behavioural function
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        # save episode lengths
        episode_lens.append(len(states))
        
        # extract target and behavioral probabilities
        target_probs = target_policy.get_probs(states, actions)
        behavioral_probs = behavior_policy.get_probs(states, actions)
        
        # initialize the return and the weight
        G = 0
        W = 1
        
        # loop backwards over the trajectory
        for timestep in range(len(states)-1, -1, -1):            
            # extract info of current timestep from trajectory    
            s = states[timestep]
            r = rewards[timestep]
            a = actions[timestep]
            G = discount_factor * G + r
            
            # add W to the sum of weights C
            C[s][a] += W
            
            # update Q function incrementally
            Q[s][a] += W/C[s][a] * (G - Q[s][a])
            
            # update the weight
            W *= (target_probs[timestep])/(behavioral_probs[timestep])
            
            # break out of the loop if the weights are 0
            if W == 0:
                break
    
    Q = Qdefaultdict2array(Q, env.nA, env.nS)     
    
    return Q, episode_lens


# ## Performance
# Plot the episode length over training

# +
# Reproducible
np.random.seed(42)

# set other parameters
epsilon = 0.05
discount_factor = 1.0
num_episodes = 10
Q = np.zeros((env.nS, env.nA))
# behavioral_policy = RandomPolicy(env.nS, env.nA)
behavioral_policy = EpsilonGreedyPolicy(Q, epsilon)
target_policy = GreedyPolicy(Q)

# the episode length is equal to the negative return. 
print(f"Updating Q using ordinary importance sampling ({num_episodes} episodes)")
Q_mc_ordinary, mc_ordinary_epslengths = mc_ordinary_importance_sampling(env, behavioral_policy, target_policy, 
                                                                        num_episodes, discount_factor, epsilon=epsilon)
# print(f"Updating Q using weighted importance sampling ({num_episodes} episodes)")
# Q_mc_weighted, mc_weighted_epslengths = mc_weighted_importance_sampling(env, behavioral_policy, target_policy,
#                                                                         num_episodes, discount_factor, epsilon=epsilon)
# -

# ## Plotting

# +
def running_mean(vals, n=1):
    assert n < len(vals)
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n 

n = 5

plt.plot(running_mean(mc_ordinary_epslengths, n), label="ordinary")
plt.plot(running_mean(mc_weighted_epslengths, n), label="weighted")
plt.title('Episode lengths MC')
plt.legend()
# plt.gca().set_ylim([0, 100])
plt.show()


# -

# ## Temporal Difference
#
# TO-DO: Make N-step

def sarsa_ordinary_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: Off-policy TD control. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keep track of useful statistics
    stats = []
    
    Q = defaultdict(lambda: defaultdict(float))
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
            
        s = env.reset()
        a = behavior_policy.sample_action(s)
        
        while True:
            # take action
            s_prime, r, final_state, _ = env.step(a)
            
            # sample action at state s_prime
            a_prime = behavior_policy.sample_action(s_prime)

            W = (target_policy.get_probs([s_prime],[a_prime])[0])/(behavior_policy.get_probs([s_prime],[a_prime])[0])
#             print(W)

            # update Q 
            Q[s][a] += alpha * W * (r + discount_factor * Q[s_prime][a_prime] - Q[s][a])    
            
            # update current s and a for next iteration
            s = s_prime
            a = a_prime
            
            R += r
            i += 1 
            
            # if final state, terminate loop
            if final_state:
                break
            
        stats.append((i, R))
        
    Q = Qdefaultdict2array(Q, env.nA, env.nS)
        
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)


# +
# Reproducible
np.random.seed(42)

# set other parameters
epsilon = 0.05
discount_factor = 1.0
num_episodes = 50
alpha=0.5
Q = np.zeros((env.nS, env.nA))
behavioral_policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)
target_policy = GreedyPolicy(Q)

# the episode length is equal to the negative return. 
print(f"Updating Q using ordinary importance sampling ({num_episodes} episodes)")
Q_td_ordinary, td_ordinary_epsstats = sarsa_ordinary_importance_sampling(env, behavioral_policy, target_policy, 
                                                                        num_episodes, discount_factor, alpha)


# +
def running_mean(vals, n=1):
    assert n < len(vals)
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n 

n = 5

plt.plot(running_mean(td_ordinary_epsstats[0], n), label="ordinary")
plt.title('Episode lengths MC')
plt.legend()
# plt.gca().set_ylim([0, 100])
plt.show()
# -

Q_td_ordinary

# ### TO-DO: TD Weighted Importance Sampling (same as above but weighted)

# +
## TD weighted importance sampling
# -

# ## Experiments


