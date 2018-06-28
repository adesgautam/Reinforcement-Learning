
import gym
import numpy as np
import pandas as pd

env = gym.make('MountainCar-v0')

class QTable:

	def __init__(self):
		self.actions = [0, 1, 2]
		self.epsilon = 0.5
		self.lr_rate = 0.01
		self.gamma = 0.9
		self.Q = pd.DataFrame(columns=self.actions, dtype=np.float64)

	def choose_action(self, state):
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
			self.Q = self.Q.append(pd.Series([0,0,0], index=self.Q.columns, name=state))

	def discretize(self, state):
		t=[]
		for i in state:
			t.append(round(i,1))
		state = str(t)
		return state

	def learn(self, state, state2, reward, action):

		# Get value from state2
		self.check_state_exist(state2)

		predict = self.Q.loc[state][action]
		target = reward + self.gamma * self.Q.loc[state2, :].max()

		self.Q.loc[state, action] += self.lr_rate * (target - predict)


# Start

QT = QTable()

state = env.reset()
state = QT.discretize(state)

for episode in range(1, 500):
	done = False
	G, reward = 0, 0

	state = QT.discretize(env.reset())

	t = 0
	print("Episode: ", episode)
	while True:
		# print(t)
		env.render()

		QT.check_state_exist(state)
		action = QT.choose_action(state)  # 1

		state2, reward, done, info = env.step(action)  # 2

		state2 = QT.discretize(state2)
		QT.learn(state, state2, reward, action)
		# print(state, action, reward)
		state = state2

		t += 1
		G += reward

		if done:
			break

	if episode % 10 == 0:
		print('Episode {0} Total Reward: {1}'.format(episode, G))
		print("Q Table:")
		for i in QT.Q:
			print(i)






