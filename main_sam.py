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
import pandas as pd
import plotly.graph_objects as go
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

# ## Environment

# +
from windy_gridworld import WindyGridworldEnv
env = WindyGridworldEnv()
# env??

# import gym
# env = gym.envs.make("FrozenLake-v0")
# env.env.__init__(is_slippery=False)
# -

try:
    env.nA = env.env.nA
    env.nS = env.env.nS
except:
    pass


# ## Policy

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


# ## Monte Carlo

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

# +
# check the length of episodes that are generated for eps greedy policy
Q = np.zeros((env.nS, env.nA))
bp = EpsilonGreedyPolicy(Q, epsilon=0.1)

for episode in range(1):
    trajectory_data = sample_episode(env, bp)
#     print("Episode {}:\nStates {}\nActions {}\nRewards {}\nDones {}\n".format(episode,*trajectory_data))
    print(f"length of episode {episode}: {len(trajectory_data[0])}")


# -

# ## MC Ordinary Importance Sampling

def Qdefaultdict2array(Q, nA, nS):
    Q_np = np.zeros((nS, nA))
    for S in range(nS):
        for A in range(nA):
            Q_np[S][A] = Q[S][A]
    return Q_np


def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, weighted=False, discount_factor=1.0,
                           sampling_function=sample_episode, epsilon=0.05, seed=42, 
                           analyse_states=[(0,2), (0,1), (14,2), (2,1), (8,2)]):
    """
    Monte Carlo prediction algorithm. Calculates the Q function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        weighted: Boolean flag to use weighted or ordinary importance sampling.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from (state, action) -> value.
    """

    # set the current Q to a large negative value
    Q = np.zeros((env.nS, env.nA))
    if weighted:
        C = np.zeros((env.nS, env.nA))
    else:
        returns_count = defaultdict(lambda: defaultdict(float))
    
    episode_lens = []
    
    # set seed
    np.random.seed(seed)
    
    analysis_values = dict((k,[]) for k in analyse_states)
    
    # sample episodes
    for i in tqdm(range(num_episodes), position=0):
        # update behavioral and target policy
        behavior_policy.Q = Q
        target_policy.Q = Q        
    
        # sample episode with new behavioural function
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        # save the episode length
        episode_lens.append(len(states)) 

        G = 0        
        W = 1
        
        # loop backwards over the trajectory
        for i, timestep in enumerate(range(len(states)-1, -1, -1)):
            s = states[timestep]
            r = rewards[timestep]
            a = actions[timestep]
            G = discount_factor * G + r
                        
            if weighted:
                # add W to the sum of weights C
                C[s][a] += W
                Q[s][a] += W/C[s][a] * (G - Q[s][a])
            else:
                returns_count[s][a] += 1 
                # use every visit incremental method
                Q[s][a] += 1/returns_count[s][a] * W * (G - Q[s][a])

            W *= (target_policy.get_probs(s, a)) / (behavior_policy.get_probs(s, a))        

            if W == 0:
                break

        # store state values to analyse
        for (s,a) in analyse_states:
#             print(Q[s][a])
            analysis_values[(s,a)].append(Q[s][a])
            
    return Q, episode_lens, analysis_values


# ## Performance

# +
# Reproducible
seed = 42

# set other parameters
epsilon = 0.1
gamma = 0.99
num_episodes = 1000
Q = np.zeros((env.nS, env.nA))
behavioral_policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)
target_policy = GreedyPolicy(Q)

# the episode length is equal to the negative return. 
print(f"Updating Q using ordinary importance sampling ({num_episodes} episodes)")
Q_mc_ordinary, mc_ordinary_epslengths, mc_analysis_ordinary = mc_importance_sampling(env,
                                                               behavioral_policy, target_policy, 
                                                               num_episodes, weighted=False,discount_factor=gamma, 
                                                               epsilon=epsilon, seed=seed)

print(f"Updating Q using weighted importance sampling ({num_episodes} episodes)")
Q_mc_weighted, mc_weighted_epslengths, mc_analysis_weighted = mc_importance_sampling(env,
                                                               behavioral_policy, target_policy,
                                                               num_episodes, weighted=True, discount_factor=gamma, 
                                                               epsilon=epsilon, seed=seed)


# +
# check how long an episode takes under the found Q function
mc_greedy_ordinary = GreedyPolicy(Q_mc_ordinary)
mc_greedy_weighted = GreedyPolicy(Q_mc_weighted)

mc_ordinary_episode = sample_episode(env, mc_greedy_ordinary)
mc_weighted_episode = sample_episode(env, mc_greedy_weighted)

print(f"resulting episode length ordinary: {len(mc_ordinary_episode[0])}")
print(f"resulting episode length weighted: {len(mc_weighted_episode[0])}")


# -

# ## Plotting

# +
def running_mean(vals, n=1):
    assert n < len(vals)
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n 

# set smoothing factor
n = 5

plt.plot(running_mean(mc_ordinary_epslengths, n), label="ordinary")
plt.plot(running_mean(mc_weighted_epslengths, n), label="weighted")
# plt.hlines(num_episodes)
plt.title('Episode lengths MC')
# plt.yscale("log")
plt.legend()
# plt.gca().set_ylim([0, 100])
plt.show()


# -

# ## Temporal Difference
#
# To-Do: Check n-step implementation so we can drop the first sarsa

# +
# def sarsa_ordinary_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0, alpha=0.5):
#     """
#     SARSA algorithm: Off-policy TD control. Calculates the value function
#     for a given target policy using behavior policy and ordinary importance sampling.
    
#     Args:
#         env: OpenAI environment.
#         policy: A policy which allows us to sample actions with its sample_action method.
#         Q: Q value function, numpy array Q[s,a] -> state-action value.
#         num_episodes: Number of episodes to run for.
#         discount_factor: Gamma discount factor.
#         alpha: TD learning rate.
        
#     Returns:
#         A tuple (Q, stats).
#         Q is a numpy array Q[s,a] -> state-action value.
#         stats is a list of tuples giving the episode lengths and returns.
#     """
    
#     # Keep track of useful statistics
#     stats = []
    
#     Q = np.zeros((env.nS, env.nA))
    
#     for i_episode in tqdm(range(num_episodes)):
#         i = 0
#         R = 0
        
#         behavior_policy.Q = Q
#         target_policy.Q = Q
            
#         s = env.reset()
#         a = behavior_policy.sample_action(s)
        
#         while True:
#             # Take action
#             s_prime, r, final_state, _ = env.step(a)
            
#             # Sample action at from next state
#             a_prime = behavior_policy.sample_action(s_prime)
            
#             # Update weight
#             W = (target_policy.get_probs([s_prime],[a_prime]))/(behavior_policy.get_probs([s_prime],[a_prime]))

#             # Update Q 
#             Q[s][a] += alpha * W * (r + discount_factor * Q[s_prime][a_prime] - Q[s][a])    
            
#             s = s_prime
#             a = a_prime
            
#             R += r
#             i += 1 
            
#             if final_state:
#                 break
            
#         stats.append((i, R))
                
#     episode_lengths, episode_returns = zip(*stats)
#     return Q, (episode_lengths, episode_returns)

# +
def sarsa_importance_sampling(env, behavior_policy, target_policy, num_episodes, weighted=False, discount_factor=1.0, alpha=0.5):
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
    num_steps = []
    
    Q = np.zeros((env.nS, env.nA))
    C = np.zeros((env.nS, env.nA))
    Q_dict = {}
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        W = 1
        
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
#             print(" ")
#             print(final_state)
#             print(W)
#             print((target_policy.get_probs([s],[a]))/(behavior_policy.get_probs([s],[a])))

#             if final_state:
#                 W = 1
            if W == 0:
                break
            
            if weighted:
#                 C[s][a] = W# + (target_policy.get_probs([s],[a]))/(behavior_policy.get_probs([s],[a]))
                Q[s][a] += alpha * (r + discount_factor * Q[s_prime][a_prime] - Q[s][a])
            else:
#                 C[s][a] = 1
                Q[s][a] += alpha * W * (r + discount_factor * Q[s_prime][a_prime] - Q[s][a]) 
#             print(C)
            
#             if C[s][a] == 0:
#                 break
                
            # Update Q 
#             Q[s][a] += alpha * W/C[s][a] * (r + discount_factor * Q[s_prime][a_prime] - Q[s][a])   
            
            
            s = s_prime
            a = a_prime
            
            i += 1 
            
            if final_state:
                break
            
        num_steps.append(i)
        Q_dict[i_episode] = Q.flatten()
                
    return Q, num_steps, Q_dict

# +
# set other parameters
epsilon = 0.05
gamma=0.99
num_episodes = 5000
alpha=0.5
Q = np.zeros((env.nS, env.nA))
behavioral_policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)
target_policy = GreedyPolicy(Q)

print(f"Updating Q using ordinary importance sampling ({num_episodes} episodes)")
Q_td_ordinary, td_ordinary_steps, _ = sarsa_importance_sampling(env, behavioral_policy, target_policy,
                                                                        num_episodes, False, gamma, alpha)

print(f"Updating Q using weighted importance sampling ({num_episodes} episodes)")
Q_td_weighted, td_weighted_steps, _ = sarsa_importance_sampling(env, behavioral_policy, target_policy,
                                                                        num_episodes, True, gamma, alpha)
# -

Q_td_ordinary

fig = go.Figure(go.Scatter(x=list(range(num_episodes)), y=td_ordinary_steps, name="ordinary"))
fig.add_trace(go.Scatter(y=td_weighted_steps, name="weighted"))
fig.update_layout(title_text="Episode lengths TD", template="plotly_white", yaxis_title="Length", xaxis_title="Number of episodes")
fig.show()

# +
# check how long an episode takes under the found Q function
greedy_ordinary = GreedyPolicy(Q_td_ordinary)
greedy_weighted = GreedyPolicy(Q_td_weighted)

ordinary_episode = sample_episode(env, greedy_ordinary)
weighted_episode = sample_episode(env, greedy_weighted)

print(f"resulting episode length ordinary: {len(ordinary_episode[0])}")
print(f"resulting episode length weighted: {len(weighted_episode[0])}")

# +
num_episodes = 5000

fig = go.Figure()
n_runs = 5
for i in range(n_runs):
    _, _, Q_dict = sarsa_importance_sampling(env, behavioral_policy, target_policy,
                                                                        num_episodes, True, gamma, alpha)
    Q_df = pd.DataFrame.from_dict(Q_dict, orient="index")
    fig.add_trace(go.Scatter(x=Q_df.index, y=Q_df[20], name="run " + str(i)))

fig.update_layout(title_text="Sarsa weighted importance sampling", template="plotly_white", yaxis_title="Q(0,0)", xaxis_title="Number of episodes")
fig.show()

# +
num_episodes = 5000

fig = go.Figure()
n_runs = 5
for i in range(n_runs):
    _, _, Q_dict = sarsa_importance_sampling(env, behavioral_policy, target_policy,
                                                                        num_episodes, False, gamma, alpha)
    Q_df = pd.DataFrame.from_dict(Q_dict, orient="index")
    fig.add_trace(go.Scatter(x=Q_df.index, y=Q_df[20], name="run " + str(i)))
    
fig.update_layout(title_text="Sarsa ordinary importance sampling", template="plotly_white", yaxis_title="Q(0,0)", xaxis_title="Number of episodes")  
fig.show()


# -

# ## N-step TD importance sampling

# +
def n_step_sarsa_ordinary_importance_sampling(env, behavior_policy, target_policy, num_episodes, n=1, discount_factor=1.0, alpha=0.5):
    """
    n-step SARSA algorithm: Off-policy TD control. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        n: number of steps
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keep track of useful statistics
    stats = []
    
#     Q = np.ones((env.nS, env.nA)) * -100
    Q = np.zeros((env.nS, env.nA))
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        behavior_policy.Q = Q
        target_policy.Q = Q
        
        s = defaultdict(lambda: defaultdict(float))
        a = defaultdict(lambda: defaultdict(float))
        r = defaultdict(lambda: defaultdict(float))
            
        s[0] = env.reset()
        a[0] = behavior_policy.sample_action(s[0])
        
        T = np.inf
        t = 0
        while True:
            if t < T:
                # Take action
                s[t+1], r[t+1], final_state, _ = env.step(a[t])
                R += r[t+1]
                i += 1
                
                if final_state:
                    T = t + 1
                else:
                    # Sample action from next state
                    a[t+1] = behavior_policy.sample_action(s[t+1])
            
            tau = t - n + 1
            
            if tau >= 0:
                # Collect states and actions included in ratio
                last_step_rho = min([tau + n, T - 1])
                first_step = tau + 1
                states = [value for key, value in s.items() if key in range(first_step, last_step_rho+1)]
                actions = [value for key, value in a.items() if key in range(first_step, last_step_rho+1)]
                
                # n-step importance sampling ratio
                rho = np.prod([(target_policy.get_probs([state],[action]))/(behavior_policy.get_probs([state],[action])) for state, action in zip(states, actions)])
                
                # n-step return
                last_step_G = min([tau + n, T])
                G = np.sum([discount_factor**(i - tau - 1) * r[i] for i in range(first_step, last_step_G)])
                if tau + n < T:
                    G += discount_factor**n * Q[s[tau+n]][a[tau+n]]
                    
                # Update Q 
                Q[s[tau]][a[tau]] += alpha * rho * (G - Q[s[tau]][a[tau]])

            if tau == T - 1:
                break
                         
            t += 1

        stats.append((i, R))
        
    Q = Qdefaultdict2array(Q, env.nA, env.nS)
        
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)


# -

def n_step_sarsa_weighted_importance_sampling(env, behavior_policy, target_policy, num_episodes, n=1, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: Off-policy TD control. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI environment.
        target policy: A policy which allows us to sample actions with its sample_action method.
        behaviour policy: A policy which allows us to sample actions with its sample_action method.
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
    
    Q = np.zeros((env.nS, env.nA))
    C = np.zeros((env.nS, env.nA))
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        behavior_policy.Q = Q
        target_policy.Q = Q
    
        s = defaultdict(lambda: defaultdict(float))
        a = defaultdict(lambda: defaultdict(float))
        r = defaultdict(lambda: defaultdict(float))

        s[0] = env.reset()
        a[0] = behavior_policy.sample_action(s[0])

        T = np.inf
        t = 0
        while True:
            if t < T:
                # Take action
                s[t+1], r[t+1], final_state, _ = env.step(a[t])
                R += r[t+1]
                i += 1

                if final_state:
                    T = t + 1
                else:
                    # Sample action from next state
                    a[t+1] = behavior_policy.sample_action(s[t+1])

            tau = t - n + 1

            if tau >= 0:
                # Collect states and actions included in ratio
                last_step_rho = min([tau + n, T - 1])
                first_step = tau + 1
                states = [value for key, value in s.items() if key in range(first_step, last_step_rho+1)]
                actions = [value for key, value in a.items() if key in range(first_step, last_step_rho+1)]

                # n-step importance sampling ratio
                rho = np.prod([(target_policy.get_probs([state],[action]))/(behavior_policy.get_probs([state],[action])) for state, action in zip(states, actions)])

                # n-step return
                last_step_G = min([tau + n, T])
                G = np.sum([discount_factor**(i - tau - 1) * r[i] for i in range(first_step, last_step_G)])
                if tau + n < T:
                    G += discount_factor**n * Q[s[tau+n]][a[tau+n]]
                    
                C[s[tau]][a[tau]] += rho
#                 Q[s][a] += W/C[s][a] * (G - Q[s][a])
                # Update Q - for weigted sampling rho/rho is 1
                Q[s[tau]][a[tau]] += alpha * rho/C[s[tau]][a[tau]] * (G - Q[s[tau]][a[tau]])

            if tau == T - 1:
                break

            t += 1

        stats.append((i, R))
        
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)


# +
# Reproducible
np.random.seed(42)

# set other parameters
epsilon = 0.05
discount_factor = 0.99
num_episodes = 10
alpha=0.5
Q = np.zeros((env.nS, env.nA))
behavioral_policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)
target_policy = GreedyPolicy(Q)

n=1
print(f"Updating Q using weighted importance sampling ({num_episodes} episodes)")
Q_td_nstep_ordinary, td_nstep_ordinary_epsstats = n_step_sarsa_ordinary_importance_sampling(env, behavioral_policy, target_policy, 
                                                                        num_episodes, n, discount_factor, alpha)
print(f"Updating Q using weighted importance sampling ({num_episodes} episodes)")
Q_td_nstep_weighted, td_nstep_weighted_epsstats = n_step_sarsa_weighted_importance_sampling(env, behavioral_policy, target_policy, 
                                                                        num_episodes, n, discount_factor, alpha)

# +
# check how long an episode takes under the found Q function
greedy_weighted_nstep = GreedyPolicy(Q_td_nstep_weighted)

weighted_episode_nstep = sample_episode(env, greedy_weighted_nstep)

print(f"resulting episode length ordinary nstep: {len(weighted_episode_nstep[0])}")

# +

n = 5

rm = running_mean(td_nstep_weighted_epsstats[0], n)

# fig = go.Figure(go.Scatter(x=list(range(len(rm))), y=rm))
# fig.show()

plt.plot(rm, label="weighted")
plt.title('Episode lengths TD')
plt.legend()
# plt.gca().set_ylim([0, 100])
plt.show()
# -

# ## Experiments




