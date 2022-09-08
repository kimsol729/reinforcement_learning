# %%
import gym
import numpy as np

from mc_function import (generate_episode_from_limit_stochastic, mc_control,
                         mc_prediction)
from plot_tools import plot_blackjack_values, plot_policy
# %%

def win_rate(env,policy, num_game):
    num_win = 0
    num_draw = 0
    num_lose = 0
    for i_episode in range(num_game):
        print("GAME START>>>\n")
        state = env.reset()
        while True:
            print("NEXT>>>\n")
            print("current state :",state)
            best_action = policy[state]
            print("my action :",best_action)
            state, reward, done, info = env.step(best_action)
            print("SOOO,\nnext state : ",state)
            print("reward:",reward)
            print("done? : ",done)
            print("info",info)
            if done:
                print('End game! Reward: ', reward)
                if reward>0:
                    # print('You won :)\n') 
                    num_win+=1
                elif reward<0: 
                    # print('You lost :(\n')
                    num_lose+=1 
                else :
                    # print('You draw :|\n')
                    num_draw+=1
                break
    return (num_win+num_draw)/num_game
# %%
env = gym.make('Blackjack-v0')

# %% run exercises
"""
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

"""

"""for i in range(10): # 경험 누적하기
    ep = generate_episode_from_limit_stochastic(env)
    print(ep)"""
    
Q_prediction = mc_prediction(env, 500000, generate_episode_from_limit_stochastic)
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
         for k, v in Q_prediction.items())
plot_blackjack_values(V_to_plot)
print("plot blackjack policy\n")
plot_policy(V_to_plot)

# %%
policy_prediction = dict((k,np.argmax(v)) for k, v in Q_prediction.items())

rate = win_rate(env,policy_prediction, 10000)
print("win rate : ",rate)

# %%
# obtain the estimated optimal policy and action-value function
policy, Q = mc_control(env, 500000, 0.02)

# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())

# plot the state-value function
print("plot blackjack value funtion\n")
plot_blackjack_values(V)
print("plot blackjack policy\n")
plot_policy(policy)

rate = win_rate(env,policy, 10000)
print("win rate : ",rate)
