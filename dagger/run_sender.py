#!/usr/bin/env python

import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from env.sender import Sender
from models import DaggerLSTM
from helpers.helpers import normalize, one_hot, softmax
import logging
import os

class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars):
        self.aug_state_dim = state_dim + action_cnt
        self.action_cnt = action_cnt
        self.prev_action = action_cnt - 1

        with tf.variable_scope('global'):
            self.model = DaggerLSTM(
                state_dim=self.aug_state_dim, action_cnt=action_cnt)

        self.lstm_state = self.model.zero_init_state(1)

        self.sess = tf.Session()

        logging.basicConfig(level=logging.WARNING, filename="/home/zyk/state.log")
        self.logger = logging.getLogger("state")

        # restore saved variables
        saver = tf.train.Saver(self.model.trainable_vars)
        saver.restore(self.sess, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables())
        uninit_vars -= set(self.model.trainable_vars)
        self.sess.run(tf.variables_initializer(uninit_vars))

    def sample_action(self, state):
        norm_state = normalize(state)

        one_hot_action = one_hot(self.prev_action, self.action_cnt)
        aug_state = norm_state + one_hot_action
        #  debug
        # print("entry")
        # print("dagger-runsender: aug_state: " + str(aug_state))
        #  debug
        # Get probability of each action from the local network.
        pi = self.model
        feed_dict = {
            pi.input: [[aug_state]],
            pi.state_in: self.lstm_state,
        }
        #debug
        self.logger.warning("RUN_SENDER: aug_state is: "+str(aug_state))
        #debug
        ops_to_run = [pi.action_probs, pi.state_out]
        action_probs, self.lstm_state = self.sess.run(ops_to_run, feed_dict)

        # Choose an action to take
        action = np.argmax(action_probs[0][0])
        self.prev_action = action
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    sender = Sender(args.port, debug=args.debug)

    model_path = path.join(project_root.DIR, 'dagger', 'model', 'model')

    learner = Learner(
        state_dim=Sender.state_dim,
        action_cnt=Sender.action_cnt,
        restore_vars=model_path)

    sender.set_sample_action(learner.sample_action)

    try:
        sender.handshake()
        sender.run()
    except KeyboardInterrupt:
        pass
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
