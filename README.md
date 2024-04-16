# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.
### States
5 Terminal States:

G (Goal): The state the agent aims to reach.

H (Hole): A hazardous state that the agent must avoid at all costs.

11 Non-terminal States:

S (Starting state): The initial position of the agent.

Intermediate states: Grid cells forming a layout that the agent must traverse.
### Actions
The agent has 4 possible actions:

0: Left
1: Down
2: Right
3: Up
### Transition Probabilities
Slippery surface with a 33.3% chance of moving as intended and a 66.6% chance of moving in orthogonal directions. For example, if the agent intends to move left, there is a

33.3% chance of moving left, a
33.3% chance of moving down, and a
33.3% chance of moving up.
### Rewards
The agent receives a reward of 1 for reaching the goal state, and a reward of 0 otherwise.

## MONTE CARLO CONTROL ALGORITHM
1.Initialize the state value function V(s) and the policy π(s) arbitrarily.

2.Generate an episode using π(s) and store the state, action, and reward sequence.

3.For each state s appearing in the episode:

G ← return following the first occurrence of s
Append G to Returns(s)
V(s) ← average(Returns(s))
4.For each state s in the episode:

π(s) ← argmax_a ∑_s' P(s'|s,a)V(s')
5.Repeat steps 2-4 until the policy converges.

6.Use the function decay_schedule to decay the value of epsilon and alpha.

7.Use the function gen_traj to generate a trajectory.

8.Use the function tqdm to display the progress bar.

9.After the policy converges, use the function np.argmax to find the optimal policy. The function takes the following arguments:

Q: The Q-table.
axis: The axis along which to find the maximum value
## MONTE CARLO CONTROL FUNCTION
pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk
import warnings ; warnings.filterwarnings('ignore')

import gym, gym_walk
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)
env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
P
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
        def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
        def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)
    
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy.

## RESULT:

Write your result here
