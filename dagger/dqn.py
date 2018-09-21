import numpy as np
import pandas as pd
import tensorflow as tf
import env.sender

np.random.seed(1)
tf.set_random_seed(1)

"""
Encapsulate DQN for global network and local network.
"""


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=2000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        # Modify : add trainable_vars
        self.target_trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_net')

        self.eval_trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net')

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # random delete rate
        self.add_new_rate = 0.9
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        """
        Modify : add op to copy network parameters from 
        """
        # self.injected_leader_variables = None
        # injected_leader_variables can't be None when constructing the computing graph

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # self.sess.run(tf.global_variables_initializer())
        # self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build  ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            # TODO : e1 to LSTM
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope(tf.get_variable_scope().name):
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t1')
                self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='t2')

            with tf.variable_scope('q_target'):
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
                self.q_target = tf.stop_gradient(q_target)
            with tf.variable_scope('q_eval'):
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            with tf.variable_scope('train'):
                self._train_op = tf.train.AdadeltaOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        done = False
        transition = np.hstack((s, [a, r], s_))

        # if len(transition == 27):
        #     print "dimension of state: %d \n action: %d \n reward: %d\n s': %d" % (len(s),1,1,len(s_))

        while not done:
            if np.random.uniform() < self.add_new_rate:
                # replace the old memory with new memory
                index = self.memory_counter % self.memory_size
                self.memory[index, :] = transition
                self.memory_counter += 1
                done = True
            else:  # save previous data, try next time
                self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]  # [...] -> [[...]]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            with tf.Session() as sess:
                # actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                actions_value = sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def sample_action(self, state):
        """
        :param state:
        :return:
        """
        # todo : remove previous line
        with tf.Session() as sess:
            action_value = sess.run(self.q_eval, feed_dict={self.s: state})
        action = np.argmax(action_value)
        return action

    def learn(self, training_batch_size=None):
        """
        Note : Workers must copy global params from leader's global network to dagger workers
        """

        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            with tf.Session() as sess:
                sess.run(self.target_replace_op)
            # self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        """ Modify : if training batch size o 
        """
        batch_size = self.batch_size
        if training_batch_size is not None:
            batch_size = training_batch_size

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=batch_size)
        batch_memory = self.memory[sample_index, :]

        # _, cost = self.sess.run(
        with tf.Session() as sess:
            _, cost = sess.run(
                [self._train_op, self.loss],
                feed_dict={
                    self.s: batch_memory[:, :self.n_features],
                    self.a: batch_memory[:, self.n_features],
                    self.r: batch_memory[:, self.n_features + 1],
                    self.s_: batch_memory[:, -self.n_features:],
                })

        # self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return cost

    def replace_target_network_param(self, network_parameters):
        self.injected_leader_variables = network_parameters
        if not hasattr(self, "inject_replace_op"):
            with tf.variable_scope('soft_replacement'):
                self.inject_replace_op = [tf.assign(t, e) for t, e in
                                          zip(self.target_trainable_vars, self.injected_leader_variables)]

        # self.sess.run(self.inject_replace_op)
        with tf.Session() as sess:
            sess.run(self.inject_replace_op)
