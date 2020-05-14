
import gym
import numpy as np
import time, pickle, os
import matplotlib.pyplot as plt


class Model():
    def __init__(self, n_states, n_actions):
        self.transitions = np.zeros((n_states, n_actions), dtype=np.uint8)
        self.rewards = np.zeros((n_states, n_actions))

    def add(self, state, action, state2, reward):
        self.transitions[state, action] = state2
        self.rewards[state, action] = reward

    def sample(self, env):
        state, action = 0, 0
        # Random visited state
        if all(np.sum(self.transitions, axis=1)) <= 0:
            state = np.random.randint(env.observation_space.n)
        else:
            state = np.random.choice(np.where(np.sum(self.transitions, axis=1) > 0)[0])

        # Random action in that state
        if all(self.transitions[state]) <= 0:
            action = np.random.randint(env.action_space.n)
        else:    
            action = np.random.choice(np.where(self.transitions[state] > 0)[0])

        return state, action

    def step(self, state, action):
        state2 = self.transitions[state, action]
        reward = self.rewards[state, action]
        return state2, reward

class Agent:
    def __init__(self, env):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.model = Model(env.observation_space.n, env.action_space.n)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state, :])

    def learn(self, state, state2, reward, action):
        # predict = Q[state, action]
        # Q[state, action] = Q[state, action] + lr_rate * (target - predict)
        target = reward + gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = (1 - lr_rate) * self.Q[state, action] + lr_rate * target

    def planning(self, n_steps):
        # if len(self.transitions)>planning_steps:
        for i in range(n_steps):
            state, action =  self.model.sample(self.env)
            state2, reward = self.model.step(state, action)
            self.learn(state, state2, reward, action)

def train():
    total_rewards = []
    for episode in range(total_episodes):
        state = agent.env.reset()
        t = 0
        ep_rewards = 0
        
        while t < max_steps:
            # env.render()

            action = agent.choose_action(state)  
            state2, reward, done, info = agent.env.step(action)  

            agent.learn(state, state2, reward, action)

            agent.model.add(state, action, state2, reward)
            agent.planning(planning_steps)

            state = state2

            ep_rewards+= reward
            t += 1
            if done:
                break
            
        total_rewards.append(ep_rewards)
        # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode) 
        # time.sleep(0.1)

        # if done:
        #     break
    return total_rewards

def test():
    total_rewards = []
    for episode in range(total_episodes):
        ep_rewards = 0
        state = agent.env.reset()
        for t in range(max_steps):
            agent.env.render()
            time.sleep(0.5)
            act = np.argmax(agent.Q[state,:])
            state2, reward, done, info = agent.env.step(act)
            if done:
                ep_rewards += reward
                break
            else:
                state = state2
        total_rewards.append(ep_rewards)
    return total_rewards 
    
def train_details(total_rewards):
    print("Q table:\n", agent.Q)
    print("Model Transitions:\n", agent.model.transitions)

    # Perfect actions: [1 2 1 0 1 0 1 0 2 1 1 0 0 2 2 0]
    print("Total Rewards in training: {0} in {1} episodes".format(sum(total_rewards), total_episodes))


def test_details(total_rewards):
    print("\n\nTotal Rewards in testing: {0} in {1} episodes".format(sum(total_rewards), total_episodes))


# Setup
env = gym.make('Taxi-v3')

epsilon = 0.9
lr_rate = 0.1
gamma = 0.99
planning_steps = 0

total_episodes = 10000
max_steps = 100

agent = Agent(env)

print("Total States:", env.observation_space.n)
print("Total Actions:", env.action_space.n)

print("####### Training ########")
total_rewards = train()
train_details(total_rewards)

print("####### Testing ########")
total_test_rewards = test()
test_details(total_test_rewards)

# with open("frozenLake_qTable.pkl", 'wb') as f:
#     pickle.dump(Q, f)








