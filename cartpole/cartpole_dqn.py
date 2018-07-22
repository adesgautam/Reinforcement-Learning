
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

env = gym.make('CartPole-v0')

class Brain():
	def __init__(self):
		self.lr_rate = 0.001
		self.epsilon = 0.9
		self.epsilon_max = 0.9
		self.epsilon_increment = 0.01
		self.reward_decay = 0.01
		self.gamma = 0.9
		self.replace_targ_iter = 300
		self.mem_size = 1000
		self.batch_size = 128
		self.nb_actions = 2
		self.n_features = 4
		self.nb_features = self.n_features*2+2 # (4 in state, 1 rew, 1 action, 4 in state_)

		# learning step counter
		self.learn_step_counter = 0
		# initialize memory
		self.memory = np.zeros((self.mem_size, self.nb_features))

		# build model
		self.build_model()

		t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
		e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

		with tf.variable_scope('soft_replacement'):
			self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		self.sess = tf.Session()
		tf.summary.FileWriter('./graph/cartpole_dqn', self.sess.graph)
		self.sess.run(tf.global_variables_initializer())	
		self.cost_his = []

	def build_layers(self, state):
		W1 = tf.Variable(tf.random_uniform([4,16], -1.0, 1.0), name="W1")
		W2 = tf.Variable(tf.random_uniform([16,2], -1.0, 1.0), name="W2")
		b1 = tf.Variable(tf.random_uniform([16], -1.0, 1.0), name="b1")
		b2 = tf.Variable(tf.random_uniform([2], -1.0, 1.0), name="b2")

		layer_1 = tf.nn.relu(tf.matmul(state, W1) + b1, name="layer1")
		predicted = tf.add(tf.matmul(layer_1, W2), b2, name="layer2")

		# w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

		# l1 = tf.layers.dense(state, 16, tf.nn.relu, kernel_initializer=w_initializer,
		# 							bias_initializer=b_initializer, name='l1')
		# predicted = tf.layers.dense(l1, 2, kernel_initializer=w_initializer,
		# 							bias_initializer=b_initializer, name='q')
		return predicted


	def build_model(self):
		self.ntwrk_state  = tf.placeholder(tf.float32, [None,self.n_features], name="state")
		self.ntwrk_state_ = tf.placeholder(tf.float32, [None,self.n_features], name="state_")
		self.ntwrk_reward = tf.placeholder(tf.float32, [None,], name="reward")
		self.ntwrk_action = tf.placeholder(tf.int32, [None,], name="action")
		
		# build layers of eval_net (changes in network, prediction)
		with tf.variable_scope('eval_net'):
			self.predicted_eval = self.build_layers(self.ntwrk_state)
		
		# build layers of target_net (target)
		with tf.variable_scope('target_net'): 
			self.predicted_target = self.build_layers(self.ntwrk_state_)

		# cal prediction
		with tf.variable_scope('q_eval'):
			# self.eval = tf.reduce_sum(tf.multiply(self.predicted_eval, self.ntwrk_action), reduction_indices=1)
			a_indices = tf.stack([tf.range(tf.shape(self.ntwrk_action)[0], dtype=tf.int32), self.ntwrk_action], axis=1)
			self.eval = tf.gather_nd(params=self.predicted_eval, indices=a_indices)    # shape=(None, )

		# calc target
		with tf.variable_scope('q_target'):
			target = self.ntwrk_reward + self.gamma * tf.reduce_max(self.predicted_target, reduction_indices=1, name="Qmax")
			self.target = tf.stop_gradient(target)

		# loss function
		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.eval, name='TD_error'))

		# optimizer
		with tf.variable_scope('train'):
			self.taining = tf.train.RMSPropOptimizer(self.lr_rate).minimize(self.loss)

		merged_summary = tf.summary.merge_all()

		sess = tf.InteractiveSession()
		summary_writer = tf.summary.FileWriter('trainsummary',sess.graph)
		sess.run(tf.global_variables_initializer())

	def learn(self):
		# replace targets
		if self.learn_step_counter % self.replace_targ_iter == 0:
			self.sess.run(self.target_replace_op)
			print("\ntarget replaced")
		# choose from batch
		if self.memory_counter > self.mem_size:
			idx = np.random.choice(self.mem_size, size=self.batch_size)
		else:
			idx = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_mem = self.memory[idx, :]

		_, cost = self.sess.run([self.loss, self.target_replace_op],
									feed_dict={ 
										self.ntwrk_state:  batch_mem[:, :self.n_features],
										self.ntwrk_action: batch_mem[:, self.n_features],
										self.ntwrk_reward: batch_mem[:, self.n_features+1],
										self.ntwrk_state_: batch_mem[:, -self.n_features:]
								})
		self.cost_his.append(cost)

		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1

	def store_transition(self, state, action, reward, state_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((state, action, reward, state_))
		idx = self.memory_counter % self.mem_size
		self.memory[idx, :] = transition
		self.memory_counter += 1


	def choose_action(self, state):
		state = state[np.newaxis, :]
		# action=0
		if np.random.random() < self.epsilon:
			action = env.action_space.sample()
		else:
			action_val = self.sess.run(self.predicted_eval, feed_dict={self.ntwrk_state: state})
			action = np.argmax(action_val)
		return action

	def plot_cost(self):
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()

brain = Brain()

step = 0
for episode in range(1, 1000):

	state = env.reset()
	print("Episode: ", episode)
	while True:
		env.render()

		# take action using epsilon-greedy
		action = brain.choose_action(state)  

		state_, reward, done, info = env.step(action)  

		# Store transition in replay memory
		brain.store_transition(state, action, reward, state_)
		# print(brain.memory)

		if step > 200:
			brain.learn()

		state = state_

		if done:
			break

		step+=1

print("GAME OVER !!!")
brain.plot_cost()





















