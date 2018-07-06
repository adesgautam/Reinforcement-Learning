
import gym
import numpy as np
import pandas as pd

env = gym.make('CartPole-v0')

class QTable:

	def __init__(self):
		self.actions = [0, 1]
		self.epsilon = 0.9
		self.lr_rate = 0.5
		self.gamma = 0.9
		self.Q = pd.DataFrame(columns=self.actions, dtype=np.float64)

	def choose_action(self, state):
		self.check_state_exist(state)

		action=0
		if np.random.random() < self.epsilon:
			action = np.random.choice(self.actions)
		else:
			action = self.Q.loc[state, :]
			action = action.reindex(np.random.permutation(action.index))  # some actions have same value
			action = action.idxmax()

		return action

	def check_state_exist(self, state):
		# Check state exists in Q table. if not exist then add to Q
		if state not in self.Q.index:
			actions = [0 for _ in range(len(self.actions))]
			self.Q = self.Q.append(pd.Series(actions, index=self.Q.columns, name=state))

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

max_r = 0
for episode in range(1, 500):
	G, reward = 0, 0

	state = QT.discretize(env.reset())

	t = 0
	while True:
		env.render()

		action = QT.choose_action(state)  

		state2, reward, done, info = env.step(action)  

		state2 = QT.discretize(state2)
		QT.learn(state, state2, reward, action)

		state = state2

		t += 1
		G += reward

		if done:
			break
	if t%20 == 0:
		QT.epsilon -= 0.1

	if G > max_r:
		max_r = G
	# if episode % 10 == 10:
	print('Episode {0} Epsilon {1} Total Reward: {2} Max Reward: {3}'.format(episode, QT.epsilon, G, max_r))






