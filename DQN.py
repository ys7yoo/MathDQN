import numpy as np
import json
import random
from collections import deque
from env import *
from config import *
import tensorflow as tf
import json
import sys
import os

GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
EPISODE = 40000
STEP = 5

VALIDATE_FREQ = 20  # validation frequency


class DQN():
    def __init__(self, env):
        self.replay_buffer = deque()
        self.good_buffer = {}
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.feat_dim
        self.action_op_dim = 3
        self.create_Q_network()
        self.create_training_method()
        self.count = 0

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.01)
        self.session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.session.run(tf.compat.v1.global_variables_initializer())

    def create_Q_network(self):
        self.state_input = tf.compat.v1.placeholder("float",[None,self.state_dim])

        W1 = self.weight_variable([self.state_dim,50])
        b1 = self.bias_variable([50])
        h_layer_1 = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

        W2 = self.weight_variable([50, 50])
        b2 = self.bias_variable([50])
        h_layer_2 = tf.nn.relu(tf.matmul(h_layer_1, W2) + b2)

        W_action_op = self.weight_variable([50, self.action_op_dim])
        b_action_op = self.bias_variable([self.action_op_dim])

        self.Q_op_value = tf.matmul(h_layer_2, W_action_op) + b_action_op

    def create_training_method(self):
        self.action_op_input = tf.compat.v1.placeholder("float",[None,self.action_op_dim]) # one hot presentation
        self.y_op_input = tf.compat.v1.placeholder("float",[None])
        self.Q_op_action = tf.reduce_sum(input_tensor=tf.multiply(self.Q_op_value,self.action_op_input),axis = 1)
        self.op_cost = tf.reduce_mean(input_tensor=tf.square(self.y_op_input - self.Q_op_action))

        self.op_optimizer = tf.compat.v1.train.AdamOptimizer(0.0001).minimize(self.op_cost)

    def perceive(self, state, action_op, reward, next_state, done, step):
        self.count += 1
        one_hot_op_action = np.zeros(self.action_op_dim)
        one_hot_op_action[action_op] = 1
        if reward > 0:
            self.good_buffer[(step, reward)] = (state, one_hot_op_action, reward, next_state, done, step)
        if self.count % 10000 == 0:
            self.count = 0
            for k, v in list(self.good_buffer.items()):
                self.replay_buffer.append(v) 
                if len(self.replay_buffer) > REPLAY_SIZE:
                    self.replay_buffer.popleft()
        else:
            self.replay_buffer.append((state, one_hot_op_action, reward, next_state, done, step))
            if len(self.replay_buffer) > REPLAY_SIZE:
                self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def egreedy_action(self, state):
        Q_op_value = self.Q_op_value.eval(feed_dict={
            self.state_input:np.array([state])
            })[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_op_dim - 1)
        else:
            return np.argmax(Q_op_value)
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
        else:
            self.epsilon = FINAL_EPSILON

    def action(self, state):
        Q_op_value = self.Q_op_value.eval(feed_dict={
            self.state_input:np.array([state])
            })[0]
        return np.argmax(Q_op_value)

    def weight_variable(self, shape):
        initial = tf.random.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def train_Q_network(self):
        # Step 1: obtain random minibatch from replay memory
        mini_batch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in mini_batch]
        action_op_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state_batch = [data[3] for data in mini_batch]

        # Step 2: calculate y
        y_op_batch = []
        Q_op_value_batch = self.Q_op_value.eval(feed_dict={self.state_input : next_state_batch})
        #print "Q_value_batch:", Q_value_batch
        for i in range(0, BATCH_SIZE):
            done = mini_batch[i][4]
            if done:
                y_op_batch.append(reward_batch[i])
            else:
                y_op_batch.append(reward_batch[i] + GAMMA * np.max(Q_op_value_batch[i]))

        #print y_batch
        #print self.Q_action.eval(feed_dict={self.action_input:action_batch, self.state_input:state_batch})
        #print self.cost.eval(feed_dict = {self.y_input:y_batch, self.action_input:action_batch,self.state_input:state_batch})
        self.op_loss = self.op_cost.eval(feed_dict={
            self.y_op_input : y_op_batch,
            self.action_op_input : action_op_batch,
            self.state_input : state_batch
        })
        # print(("operate_loss", self.op_loss))

        self.op_optimizer.run(feed_dict={
            self.y_op_input : y_op_batch,
            self.action_op_input : action_op_batch,
            self.state_input : state_batch
        })


def main():
    config = Config()
    config.analysis_filename = config.analysis_filename + "_" + sys.argv[1]
    config.train_list, config.validate_list = config.seperate_date_set(sys.argv[1])

    print('prepare environment')
    env = Env(config)
    env.make_env()

    print('set up DQN')
    dqn = DQN(env)
    #checkpoint_dir = "./model/fold" + sys.argv[1]
    #latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    #start = int(latest_checkpoint[14:latest_checkpoint.find("_model")])+1

    max_accuracy = 0
    saver = tf.compat.v1.train.Saver()
    start = 0
    #saver.restore(dqn.session, latest_checkpoint)
    #saver = tf.train.Saver()
    reward_list = []

    print('start iterations')
    for episode in range(EPISODE)[start:]:
        total_reward = 0
        env.reset_inner_count()
        #print 'episode', episode

        for itr in range(config.train_num):
            state = env.reset()
            for step in range(STEP):
                #print(("--episode:", episode, "iter: ", itr, "step: ", step))

                action_op = dqn.egreedy_action(state)

                next_state, reward, done = env.step(action_op)
                total_reward += reward

                dqn.perceive(state, action_op, reward, next_state, done, config.train_list[env.count-1])

                state = next_state

                if done:
                    break

        reward_list.append(total_reward)

        with open("./test/reward_list_"+str(sys.argv[1])+".json", 'w') as f:
            json.dump(reward_list, f)

        if episode % VALIDATE_FREQ == 0:
            #save_path = saver.save(dqn.session, os.path.join("./model/fold"+str(sys.argv[1]),str(episode)+"_model.ckpt"))
            with open(config.analysis_filename, 'a') as f:
                f.write("test episode: "+str(episode) + '\n')

            # validate
            right_count = 0
            for itr in range(config.validate_num):
                state = env.validate_reset(itr)
                for step in range(STEP):
                    action_op = dqn.action(state)

                    next_state, done, flag, _ = env.val_step(action_op, sys.argv[1])

                    state = next_state

                    if done:
                        right_count += flag
                        break
                # print(("test_index:", config.validate_list[itr], "reward", total_reward))
            this_accuracy = right_count / config.validate_num

            # save and print validation results
            if this_accuracy > max_accuracy:
                max_accuracy = this_accuracy
                save_path = saver.save(dqn.session, os.path.join("./model/fold"+str(sys.argv[1]), str(episode)+"_model.ckpt"))
            with open("./test/test_info"+"_"+sys.argv[1]+".data", 'a') as f:
                f.write("episode:{:.0f}, correct operator:{:.0f}, acc:{:.4f},  operator_loss:{:.4f}\n".\
                         format((episode), (right_count), (this_accuracy), (dqn.op_loss)))
            print('[validation] episode {}, accuracy={:.2f}, operator_loss={:.2f}'.format(episode, this_accuracy, dqn.op_loss))


if __name__ == '__main__':
    main()  
