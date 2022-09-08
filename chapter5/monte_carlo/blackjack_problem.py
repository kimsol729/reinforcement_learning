# %%
import gym
import sys
import numpy as np
from collections import defaultdict
from plot_tools import plot_blackjack_values, plot_policy
from tqdm import tqdm

# %% load environment of blackjack
env = gym.make('Blackjack-v0')

# %% run exercises
for i_episode in range(10):
    print("GAME START>>>\n")
    state = env.reset()
    while True:
        print("NEXT>>>\n")
        print("current state :",state)
        action = env.action_space.sample()
        print("my action :",action)
        state, reward, done, info = env.step(action)
        print("SOOO,\nnext state : ",state)
        print("reward:",reward)
        print("done? : ",done)
        print("info",info)
        if done:
            print('End game! Reward: ', reward)
            if reward>0:
                print('You won :)\n') 
            elif reward<0: 
                print('You lost :(\n') 
            else :
                print('You draw :|\n')
            break
# %% 1. MC prediction
def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

# %%
for i in range(10): # 경험 누적하기
    ep = generate_episode_from_limit_stochastic(env)
    print(ep)

# %%
def mc_prediction_q_first_(env, num_episodes, generate_episode, gamma=1.0):
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for i_episode in range(1, num_episodes+1):
        episode = generate_episode(env)
        states, actions, rewards = zip(*episode)
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            returns_sum[state][actions[i]] += sum(rewards[i:]*discounts[:-(1+i)])
            N[state][actions[i]] += 1.0
            Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]
    return Q
# %%
Q = mc_prediction_q_first_(env, 50000, generate_episode_from_limit_stochastic)
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
         for k, v in Q.items())
plot_blackjack_values(V_to_plot)
print("plot blackjack policy\n")
plot_policy(V_to_plot)



# %% 2. MC control
def generate_episode_from_Q(env, Q, epsilon, nA):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = env.reset()
    while True:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s

def update_Q(env, episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]] 
        Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
    return Q

# %%
def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    for i_episode in tqdm(range(1, num_episodes+1)):
        epsilon = max(epsilon*eps_decay, eps_min)
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        Q = update_Q(env, episode, Q, alpha, gamma)
        policy = dict((k,np.argmax(v)) for k, v in Q.items())

    return policy, Q
# %%
# obtain the estimated optimal policy and action-value function
policy, Q = mc_control(env, 50000, 0.02)
# %%
# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())

# plot the state-value function
print("plot blackjack value funtion\n")
plot_blackjack_values(V)
print("plot blackjack policy\n")
plot_policy(policy)
# %%
