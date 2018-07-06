
import gym
import numpy as np
import pandas as pd
import time
import pickle, os

env = gym.make('Taxi-v2')

with open("taxi_qTable.pkl", 'rb') as f:
	Q = pickle.load(f)

def choose_action(state):
	action = np.argmax(Q[state, :])
	return action

# start
for episode in range(15):

	state = env.reset()
	print("********Episode: ", episode)
	t = 0
	while t < 100:
		env.render()

		action = choose_action(state)  
		print("Action: ", action)
		state2, reward, done, info = env.step(action)  
		# if state2==15:
		# 	print("state 15")
		state = state2

		if done:
			break

		time.sleep(0.5)
		os.system('clear')

