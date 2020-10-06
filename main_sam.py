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
    
    def get_probs(self, state, action):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """   
        # for state and action only:
        action_probs = self.Q[state]
        max_indices = np.argwhere(action_probs == np.amax(action_probs))
        
        if action in max_indices:
            prob = 1/len(max_indices)
        else:
            prob = 0
        
        return prob
        
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        best_action = np.random.choice([i for i, j in enumerate(self.Q[obs]) if j == np.max(self.Q[obs])])
        
        return best_action
        


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
        
    def get_probs(self, state, action):
        # for one state and action 
        action_probs = self.Q[state]
        max_indices = np.argwhere(action_probs == np.amax(action_probs))
        # all probs are equal, give all equal probabilities
        if len(max_indices) == len(action_probs):
            return 1/len(max_indices)
            
        if action in max_indices:
            prob = (1-self.epsilon)/len(max_indices)
        else:
            prob = epsilon / (len(action_probs) - len(max_indices))
        
        return prob
        
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        p = np.random.uniform()
        if p > self.epsilon:
            # choose one of the best actions
            action = np.random.choice([i for i, j in enumerate(self.Q[obs]) if j == np.max(self.Q[obs])])
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
for episode in range(1):
    trajectory_data = sample_episode(env, random_policy)
#     print("Episode {}:\nStates {}\nActions {}\nRewards {}\nDones {}\n".format(episode,*trajectory_data))
    print(f"length of episode {episode}: {len(trajectory_data[0])}")


# +
# check the length of episodes that are generated for eps greedy policy
Q = np.zeros((env.nS, env.nA))
bp = EpsilonGreedyPolicy(Q, epsilon=0.1)

for episode in range(1):
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
#     Q = defaultdict(lambda: defaultdict(float))
    Q = np.ones((env.nS, env.nA)) * -100
    returns_count = defaultdict(lambda: defaultdict(float))
    
    episode_lens = []
    
    # sample episodes
    for i in tqdm(range(num_episodes), position=0):
        # update behavioral function:
#         behavior_policy = EpsilonGreedyPolicy(Qdefaultdict2array(Q, env.nA, env.nS), epsilon)
        behavior_policy.Q = Q
#         target_policy = GreedyPolicy(Qdefaultdict2array(Q, env.nA, env.nS))
        target_policy.Q = Q        
    
        # sample episode with new behavioural function
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        # save the episode length
        episode_lens.append(len(states))
        
        # extract target and behavioral probabilities
#         target_probs = target_policy.get_probs(states, actions)
#         behavioral_probs = behavior_policy.get_probs(states, actions)
#         target_probs = [target_policy.get_probs(states[t], actions[t]) for t in range(len(states))]
#         behavioral_probs = [behavioral_policy.get_probs(states[t], actions[t]) for t in range(len(states))]
            

        G = 0        
        W = 1
        
        # loop backwards over the trajectory
        for i, timestep in enumerate(range(len(states)-1, -1, -1)):
#             print(i)
            s = states[timestep]
            r = rewards[timestep]
            a = actions[timestep]
            G = discount_factor * G + r
            
            returns_count[s][a] += 1 

            # compute the ratio using the two probability lists
#             ratio = np.prod([t/b for t, b in zip(target_probs[timestep:], behavioral_probs[timestep:])])
            
            # use every visit incremental method
            Q[s][a] += 1/returns_count[s][a] * (W * G - Q[s][a])
            
            W *= (target_policy.get_probs(s, a)) / (behavior_policy.get_probs(s, a))
            
            if W == 0:
                break
    
#     Q = Qdefaultdict2array(Q, env.nA, env.nS)
    
    return Q, episode_lens


# -

# ### MC: Weighted Importance Sampling
#
# ##### (TODO: Eventually: merge the two functions into one with a weighted flag)

# +
def mc_weighted_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
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

    # create a matrix defaultdict for the Q function and the sum of weights C
#     Q = defaultdict(lambda: defaultdict(float))
    Q = np.ones((env.nS, env.nA)) * -100
    C = defaultdict(lambda: defaultdict(float))
    
    episode_lens = []
    
    # sample episodes
    for i in tqdm(range(num_episodes), position=0):
        # update behavioral function:
#         behavior_policy = EpsilonGreedyPolicy(Qdefaultdict2array(Q, env.nA, env.nS), epsilon)
        behavior_policy.Q = Q
#         target_policy = GreedyPolicy(Qdefaultdict2array(Q, env.nA, env.nS))
        target_policy.Q = Q
        # sample episode with new behavioural function
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        # save episode lengths
        episode_lens.append(len(states))
        
        # extract target and behavioral probabilities OLD
#         target_probs = target_policy.get_probs(states, actions)
#         behavioral_probs = behavior_policy.get_probs(states, actions)
       
        # print(Qdefaultdict2array(Q, env.nA, env.nS))
#         print(target_probs)
        
        
        # initialize the return and the weight
        G = 0
        W = 1
        
        # loop backwards over the trajectory
        for i, timestep in enumerate(range(len(states)-1, -1, -1)): 
#             print(i)
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
            # W *= (target_probs[timestep])/(behavioral_probs[timestep])
            W *= (target_policy.get_probs(s, a)) / (behavior_policy.get_probs(s, a))
            
            # break out of the loop if the weights are 0
            if W == 0:
                break
    
#     Q = Qdefaultdict2array(Q, env.nA, env.nS)     
    
    return Q, episode_lens
# -

# ## Performance
# Plot the episode length over training

# +
# Reproducible
np.random.seed(42)

# set other parameters
epsilon = 0.1
discount_factor = 1.0
num_episodes = 5000
Q = np.ones((env.nS, env.nA)) * -100
behavioral_policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)
target_policy = GreedyPolicy(Q)

# the episode length is equal to the negative return. 
print(f"Updating Q using ordinary importance sampling ({num_episodes} episodes)")
Q_mc_ordinary, mc_ordinary_epslengths = mc_ordinary_importance_sampling(env, behavioral_policy, target_policy, 
                                                                        num_episodes, discount_factor, epsilon=epsilon)
print(f"Updating Q using weighted importance sampling ({num_episodes} episodes)")
Q_mc_weighted, mc_weighted_epslengths = mc_weighted_importance_sampling(env, behavioral_policy, target_policy,
                                                                        num_episodes, discount_factor, epsilon=epsilon)


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

# +
# check how long an episode takes under the found Q function
greedy_ordinary = GreedyPolicy(Q_mc_ordinary)
greedy_weighted = GreedyPolicy(Q_mc_weighted)

ordinary_episode = sample_episode(env, greedy_ordinary)
weighted_episode = sample_episode(env, greedy_weighted)

print(f"resulting episode length ordinary: {len(ordinary_episode[0])}")
print(f"resulting episode length weighted: {len(weighted_episode[0])}")


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
    
    Q = np.ones((env.nS, env.nA)) * -100
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        behavior_policy.Q = Q
        target_policy.Q = Q
            
        s = env.reset()
        a = behavior_policy.sample_action(s)
        
        while True:
            # Take action
            s_prime, r, final_state, _ = env.step(a)
            
            # Sample action at from next state
            a_prime = behavior_policy.sample_action(s_prime)
            
            # Update weight
            W = (target_policy.get_probs([s_prime],[a_prime]))/(behavior_policy.get_probs([s_prime],[a_prime]))

            # Update Q 
            Q[s][a] += alpha * W * (r + discount_factor * Q[s_prime][a_prime] - Q[s][a])    
            
            s = s_prime
            a = a_prime
            
            R += r
            i += 1 
            
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
num_episodes = 5000
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
plt.title('Episode lengths TD')
plt.legend()
# plt.gca().set_ylim([0, 100])
plt.show()

# +
# check how long an episode takes under the found Q function
greedy_ordinary = GreedyPolicy(Q_td_ordinary)

ordinary_episode = sample_episode(env, greedy_ordinary)

print(f"resulting episode length ordinary: {len(ordinary_episode[0])}")
# -

# ### TO-DO: TD Weighted Importance Sampling (same as above but weighted)

# +
## TD weighted importance sampling
# -

# ## Experiments


