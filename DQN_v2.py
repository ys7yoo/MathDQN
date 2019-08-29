import numpy as np
import json
import random
from collections import deque
import pprint
from env import *
from config import *
import tensorflow as tf
import json
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
from matplotlib import style

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
style.use('ggplot')

GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
EPISODE = 80000
STEP = 5

class DQN():
    def __init__(self, env):
        self.replay_buffer = deque()
        self.good_buffer =  {} 
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.feat_dim
        self.action_op_dim = 3
        self.create_Q_network()
        self.create_training_method()
        self.count = 0

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
        self.session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        self.state_input = tf.placeholder("float",[None,self.state_dim])

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
        self.action_op_input = tf.placeholder("float",[None,self.action_op_dim]) # one hot presentation
        self.y_op_input = tf.placeholder("float",[None])
        self.Q_op_action = tf.reduce_sum(tf.multiply(self.Q_op_value,self.action_op_input),reduction_indices = 1)
        self.op_cost = tf.reduce_mean(tf.square(self.y_op_input - self.Q_op_action))

        self.op_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.op_cost)

    def perceive(self,state, action_op,reward,next_state,done, step):
        self.count += 1
        one_hot_op_action = np.zeros(self.action_op_dim)
        one_hot_op_action[action_op] = 1
        if reward > 0 :
            self.good_buffer[(step,reward)] = (state,one_hot_op_action,reward,next_state,done, step)
        if self.count % 10000 == 0:
            self.count = 0
            for k,v in list(self.good_buffer.items()):
                self.replay_buffer.append(v) 
                if len(self.replay_buffer) > REPLAY_SIZE:
                    self.replay_buffer.popleft()
        else:
            self.replay_buffer.append((state,one_hot_op_action,reward,next_state,done, step))
            if len(self.replay_buffer) > REPLAY_SIZE:
                self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def egreedy_action(self,state):
        Q_op_value = self.Q_op_value.eval(feed_dict = {
            self.state_input:np.array([state])
            })[0]
        if random.random() <= self.epsilon:
            return random.randint(0,self.action_op_dim - 1)
        else:
            return np.argmax(Q_op_value)
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
        else:
            self.epsilon = FINAL_EPSILON

    def action(self,state):
        Q_op_value = self.Q_op_value.eval(feed_dict = {
            self.state_input:np.array([state])
            })[0]
        return np.argmax(Q_op_value)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def train_Q_network(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_op_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]


        # Step 2: calculate y
        y_op_batch = []
        Q_op_value_batch = self.Q_op_value.eval(feed_dict={self.state_input:next_state_batch})
        #print "Q_value_batch:", Q_value_batch
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_op_batch.append(reward_batch[i])
            else :
                y_op_batch.append(reward_batch[i] + GAMMA * np.max(Q_op_value_batch[i]))

        #print y_batch
        #print self.Q_action.eval(feed_dict={self.action_input:action_batch, self.state_input:state_batch})
        #print self.cost.eval(feed_dict = {self.y_input:y_batch, self.action_input:action_batch,self.state_input:state_batch})
        self.op_loss = self.op_cost.eval(feed_dict={
            self.y_op_input:y_op_batch,
            self.action_op_input:action_op_batch,
            self.state_input:state_batch
        })
        #print(("operate_loss", self.op_loss))
        self.op_optimizer.run(feed_dict={
            self.y_op_input:y_op_batch,
            self.action_op_input:action_op_batch,
            self.state_input:state_batch
        })

def change_action_to_op(action_op):
    if action_op == 0:
        result = '+'
    elif action_op == 1:
        result = '-'
    else:
        result = 'in-'
    return result

def real_op(node, gold_answer):
    ans = []
    for i in range(len(node)):
        node[i] = float(node[i])
    gold_answer = float(gold_answer)
    if abs(node[0] + node[1] - gold_answer) < 0.0001:
        return '+'
    if abs(node[0] - node[1] - gold_answer) < 0.0001:
        return '-'
    if abs(node[1] - node[0] - gold_answer) < 0.0001:
        return 'in-'
    if abs(sum(node) - gold_answer) < 0.0001:
        return '+, +'
    if len(node) > 2:
        if abs(node[0] + node[1] - node[2] - gold_answer) < 0.0001:
            return '+, -'
        if abs(node[0] - node[1] + node[2] - gold_answer) < 0.0001:
            return '-, +'
        if abs(node[0] - node[1] - node[2] - gold_answer) < 0.0001:
            return '-, -'
    return 'node'

def wrong_answer_case(op, ans):
    if op[0] == '-' and ans == '+': return 0
    if op[0] == 'in-' and ans == '+': return 1
    if op[0] == '+' and ans == '-': return 2
    if op[0] == 'in-' and ans == '-': return 3
    if op[0] == '+' and ans == 'in-': return 4
    if op[0] == '-' and ans == 'in-': return 5
    if ans == '+, +': return 6
    if ans == '+, -': return 7
    if ans == '-, +': return 8
    if ans == '-, -': return 9
    return 10

def main():
    test_cnt = 0
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
    saver = tf.train.Saver()
    start = 0
    #saver.restore(dqn.session, latest_checkpoint)
    #saver = tf.train.Saver()
    reward_list = []

    print('start iterations\n')
    for episode in range(EPISODE)[start:]:
        total_reward = 0
        env.reset_inner_count()

        for itr in range(config.train_num):
            state = env.reset()
            for step in range(STEP):
                action_op = dqn.egreedy_action(state)
                next_state,reward,done = env.step(action_op)
                total_reward += reward
                dqn.perceive(state, action_op, reward, next_state, done, config.train_list[env.count-1])
                state = next_state
                if done:
                    break
        reward_list.append(total_reward)

        with open("./test/reward_list_"+str(sys.argv[1])+".json", 'w') as f:
             json.dump(reward_list, f)

        if episode % 400 == 0:
            X_op = []
            X_index = []
            ans = []
            op_result = []

            done_cnt = 0
            right_count = 0
            total_result = []
            #save_path = saver.save(dqn.session, os.path.join("./model/fold"+str(sys.argv[1]),str(episode)+"_model.ckpt"))
            with open(config.analysis_filename, 'a') as f:
                 f.write("test episode: "+str(episode) + '\n')
            for itr in range(config.validate_num):
                state = env.validate_reset(itr)
                node_cnt = 0
                node = []
                op = []
                for step in range(STEP):
                    action_op = dqn.action(state)
                    next_state, done,flag, node1, node2, DQN_answer, gold_answer, _ = env.val_step(action_op, sys.argv[1])
                    state = next_state
                    if node_cnt == 0:
                        node.append(node1)
                        node.append(node2)
                    else:
                        node.append(node2)
                    op.append(change_action_to_op(action_op))
                    node_cnt += 1
                    if done:
                        done_cnt +=1
                        right_count += flag
                        act = change_action_to_op(action_op)
                        if flag:
                            #print(("test_index:", config.validate_list[itr]))
                            total_result.append("{0}번) node: {1}, DQN's op: {2}, DQN's answer: {3}, gold_answer: {4}, 맞았습니다!".format(str(config.validate_list[itr]), node, op, DQN_answer, gold_answer)) 
                        else:
                            #print(("test_index:", config.validate_list[itr]))
                            total_result.append("{0}번) node: {1}, DQN's op: {2}, DQN's answer: {3}, gold_answer: {4}, 틀렸습니다!".format(str(config.validate_list[itr]), node, op, DQN_answer, gold_answer)) 
                            ans.append(real_op(node, gold_answer))
                            X_index.append(config.validate_list[itr])
                            X_op.append(op)
                            op_result.append(wrong_answer_case(op, real_op(node, gold_answer)))
                        break

            this_accuracy = right_count*1.0/config.validate_num
            if this_accuracy > max_accuracy:
                max_accuracy = this_accuracy
                save_path = saver.save(dqn.session, os.path.join("./model/fold"+str(sys.argv[1]),str(episode)+"_model.ckpt"))
            with open("./test/test_info"+"_"+sys.argv[1]+".data", 'a') as f:
                f.write("episode:{:.0f}, correct operator:{:.0f}, acc:{:.4f},  operator_loss:{:.4f}\n".\
                         format((episode), (right_count), (right_count*1.0/config.validate_num), (dqn.op_loss)))
            print('episode:',episode,'Accuracy:' , right_count*1.0/config.validate_num)
            #print('\n문제 {0}개 중 {1}개 맞음!\n'.format(config.validate_num, right_count)) 
            print('문제 {0}개 중 {1}개 틀림!'.format(config.validate_num, config.validate_num - right_count)) 

            #print(X_index)
            #print(op_result, '\n')
            
            wrong_case_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for c in op_result:
                wrong_case_num[c] += 1
            wrong_case = ['+인데-계산', '+인데in-계산', '-인데+계산', '-인데in-계산', 'in-인데+계산', 'in-인데-계산',
                          '+, +인데 다르게 계산', '+, -인데 다르게 계산', '-, +인데 다르게 계산', '-, -인데 다르게 계산',
                          '노드 문제', '그 외의 경우']
            for k in range(len(wrong_case_num)):
                if test_cnt == 0:
                    print('{0}: {1}'.format(wrong_case[k], wrong_case_num[k]))
                else:
                    print('{0}: {1} ({2} --> {3})'.format(wrong_case[k], wrong_case_num[k], li_before[k], wrong_case_num[k]))
            test_cnt += 1
            li_before = wrong_case_num

            fig = plt.figure(figsize=(24, 15))
            ax = fig.add_subplot(111)

            pos = np.arange(12)
            rects = plt.bar(pos, wrong_case_num, align='center', width=1)
            plt.xticks(pos, wrong_case)

            for i, rect in enumerate(rects):
                ax.text(rect.get_x() + rect.get_width() / 2.0, 0.96 * rect.get_height(), str(wrong_case_num[i]) + '%',
                        ha='center')

            plt.ylabel('틀린 문제 수')
            plt.title('Accuracy: {0}'.format(right_count*1.0/config.validate_num))
            fig = plt.gcf()
            fig.savefig('image_fold_{0}/episode{1}.png'.format(sys.argv[1], episode))

            print('-'*70)

        if right_count * 1.0 / config.validate_num > 0.9:
            print('Good job DQN!')
            break

if __name__ == '__main__':
   main()  
