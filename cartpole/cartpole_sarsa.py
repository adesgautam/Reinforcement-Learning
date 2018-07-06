
import gym
import numpy as np
import pandas as pd

env = gym.make('CartPole-v0')

class SarsaTable:

	def __init__(self):
		self.actions = [0, 1]
		self.epsilon = 0.9
		self.lr_rate = 0.01
		self.gamma = 0.9
		self.Q = pd.DataFrame(columns=self.actions, dtype=np.float64)

	def choose_action(self, state):
		self.check_state_exist(state)

		action=0
		if not np.random.random() < self.epsilon:
			action = self.Q.loc[state, :]
			action = action.reindex(np.random.permutation(action.index))     # some actions have same value
			action = action.idxmax()
		else:
			action = np.random.choice(self.actions)

		return action

	def check_state_exist(self, state):
		# Check state exists in Q table. if not exist then add to Q
		if state not in self.Q.index:
			self.Q = self.Q.append(pd.Series([0,0], index=self.Q.columns, name=state))

	def discretize(self, state):
		t=[]
		for i in state:
			t.append(round(i,2))
		state = str(t[2:])
		return state

	def learn(self, state, state2, reward, action, action2):
		# Get value from state2
		self.check_state_exist(state2)

		predict = self.Q.loc[state][action]
		target = reward + self.gamma * self.Q.loc[state2, action2]

		self.Q.loc[state, action] += self.lr_rate * (target - predict)


# Start

ST = SarsaTable()

max_r = 0
for episode in range(1, 500):
	G, reward = 0, 0

	state = ST.discretize(env.reset())

	action = ST.choose_action(state)

	t = 0
	while True:
		env.render()

		state2, reward, done, info = env.step(action)  
		state2 = ST.discretize(state2)

		action2 = ST.choose_action(state2)
		
		ST.learn(state, state2, reward, action, action2)
		print(state2, reward, action2)
		
		state = state2
		action = action2

		t += 1
		G += reward

		if done:
			break

	if G > max_r:
		max_r = G
	# if episode % 10 == 10:
	print('Episode {0} Total Reward: {1} Max Reward: {2}'.format(episode, G, max_r))




























