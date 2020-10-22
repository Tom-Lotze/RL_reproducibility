# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-10-22 18:37
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-10-22 18:38

import numpy as np

def value_iter_q(env, theta=0.0001, discount_factor=0.95):
    """
    Q-value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
    """

    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))

    while True:
        delta = 0

        # update Q (value iteration)
        for state in range(env.nS):
            for action in range(env.nA):
                old_value = Q[state][action]
                action_value = 0
                # loop over transition values
                for (prob, next_state, reward, _) in env.P[state][action]:
                    # value of the next state is the maximum q value.
                    reward = float(reward)
                    next_state_value = np.max(Q[next_state])
                    action_value += prob * (reward + discount_factor * next_state_value)

                # update the action value in Q
                if action_value > 1:
                    print(action_value)
                    breakpoint()
                Q[state][action] = action_value
                # check if there were any changes in the last loop
                delta = max(delta, abs(old_value - action_value))


        if delta < theta:
            break

    return Q
