
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import matplotlib.pyplot as plt
#import os
import sys

from collections import deque

from env import Env
from config import *


NUM_STEP = 500
class DQN:
    def __init__(self, dim_state=4, dim_action=2, batch_size=32):
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.batch_size = batch_size

        self.model = keras.models.Sequential([
            keras.layers.Dense(50, activation="elu", input_shape=[dim_state]),
            keras.layers.Dense(50, activation="elu"),
            keras.layers.Dense(dim_action)
        ])
        print(self.model.summary())

        self.replay_memory = deque(maxlen=2000)

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(2)
        else:
            Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values[0])

    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.replay_memory), size=batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def play_one_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done = env.step(action)
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done

    def training_step(self, batch_size=32, discount_rate=0.95, optimizer = keras.optimizers.Adam(lr=1e-3), loss_fn = keras.losses.mean_squared_error):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences

        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)

        target_Q_values = rewards + (1 - dones) * discount_rate * max_next_Q_values

        mask = tf.one_hot(actions, self.dim_action)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)

        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    ###########################################################################
    # some helper functions
    def set_weights(self, weight):
        self.model.set_weights(weight)

    def get_weights(self):
        return self.model.get_weights()

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = keras.models.load_model(filename)

    ###########################################################################

def plot_observations(observations):
    observations = np.array(observations)

    print("standard deviations=", np.std(observations, axis=0))

    plt.subplot(211)
    plt.plot(observations[:, :2])
    plt.legend(['horizontal position', 'velocity'])

    plt.subplot(212)
    plt.plot(observations[:, 2:])
    plt.legend(['angle of the pole', 'angular velocity'])


def main():

    print('Set up the config...')
    config = Config()
    config.analysis_filename = config.analysis_filename + "_" + sys.argv[1]
    config.train_list, config.validate_list = config.seperate_date_set(sys.argv[1])

    print('prepare environment')

    # to make output stable across runs
    np.random.seed(42)
    tf.random.set_seed(42)

    env = Env(config)
    env.make_env()



    ##############################################
    ## train
    print('training...')

    # env.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    print('training...')
    dqn = train(env, config)
    dqn.save("dqn_math.h5")

    # # load saved DQN model
    # dim_state = env.feat_dim
    # dim_action = 3
    # dqn = DQN(dim_state=dim_state, dim_action=dim_action, batch_size=32)


    ## test trained DQN
    # env.seed(42)
    dqn.load("dqn_math.h5")

    print('testing...')
    for i in range(5):
        test(env, dqn)

def train(env, config):
    print('set up DQN')
    dim_state = env.feat_dim
    dim_action = 3

    dqn = DQN(dim_state=dim_state, dim_action=dim_action, batch_size=32)


    rewards = []
    best_score = 0
    for episode in range(600):
        total_reward = 0
        env.reset_inner_count()

        for itr in range(config.train_num):  # WHY DO I NEED THIS?
            obs = env.reset()
            for step in range(NUM_STEP):
                epsilon = max(1 - episode / 500, 0.01)

                obs, reward, done = dqn.play_one_step(env, obs, epsilon)

                if done:
                    break

            # keep the best model with the best score
            rewards.append(step)  # Not shown in the book
            if step > best_score:  # Not shown
                best_weights = dqn.model.get_weights()  # Not shown
                best_score = step  # Not shown
            print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="")  # Not shown

        if episode > 50:
            dqn.training_step()

    # save the best model
    dqn.set_weights(best_weights)

    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Sum of rewards", fontsize=14)
    plt.savefig("dqn_rewards.png")
    #plt.show()

    return dqn

def test(env, dqn):
    state = env.reset()
    states = []
    total_reward = 0
    # frames = []
    for step in range(200):
        action = dqn.epsilon_greedy_policy(state)

        state, reward, done = env.step(action)

        states.append(state)
        total_reward += reward
        if done:
            break

        # img = env.render(mode="rgb_array")
        # frames.append(img)
    states = np.array(states)
    np.savetxt("test_states.txt", states, fmt='%.2f')
    print("total_reward=", total_reward)
    # plot_animation(frames)

if __name__ == '__main__':
    main()
