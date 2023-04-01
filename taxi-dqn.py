import random
from collections import deque

import gym
import numpy as np
import progressbar
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Activation
from tensorflow.keras.optimizers import Adam
import os, time, sys

import logging
logging.basicConfig(
            level=logging.DEBUG,
            format="[%(filename)s - %(funcName)20s:%(lineno)s] : %(message)s ",
            handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler("log_{}.log".format(time.time()), mode="w")]
        )
logger = logging.getLogger()
logger.info("\n\n\nHey, Welcome Amitabh !\n\n\n")

envn = gym.make("Taxi-v3").env
envn.render()


logger.info(f"Observation space      : {envn.observation_space}")
logger.info(f"Number of states       : {envn.observation_space.n}")
logger.info(f"Action space           : {envn.action_space}")
logger.info(f"Number of action space : {envn.action_space.n}")


class Agent:
    def __init__(self, envn, optimizer):
        self._state_size = envn.observation_space.n
        self._action_size = envn.action_space.n
        self._optimizer = optimizer

        self.experience_replay = deque(maxlen=2000)

        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1

        # Build network
        self.q_network = self.build_compile_model()
        self.target_network = self.build_compile_model()
        self.align_target_model()  # To update the target model every update interval

    def store_experience(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))
    
    def build_compile_model(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        
        model.add(Dense(50))
        model.add(Activation('relu'))
        
        model.add(Dense(50))
        model.add(Activation('relu'))

        model.add(Dense(self._action_size))
        model.add(Activation('linear'))  # Used for a continious variable

        model.compile(loss='mse', optimizer=self._optimizer)
        logger.info(model.summary)
        dot_img_file = os.path.join(os.getcwd(), f'model_{time.time()}.png')
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        return model


    def align_target_model(self):
        logger.info(f"Weights of the network : {self.q_network.get_weights()}")
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return envn.action_space.sample()

        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):   # equal to experience_replay
        minibatch = random.sample(self.experience_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:
            target = self.q_network.predict(state)

            if terminated:
                target[0][action] = reward

            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            logger.info(f"Target Network in retrain : {target}")
            self.q_network.fit(state, target, epochs=1, verbose=0)

optimizer = Adam(learning_rate=0.01)
agent = Agent(envn, optimizer)

batch_size=32
num_of_episodes = 10
timesteps_per_episode = 1000
agent.q_network.summary()


for e in range(0, num_of_episodes):
    state = envn.reset()
    state = np.reshape(state, [1, 1])

    # Initialize rewards
    total_reward = 0
    terminated = False

    bar = progressbar.ProgressBar(maxval=timesteps_per_episode / 10,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for timestep in range(timesteps_per_episode):
        action = agent.act(state)

        # Take action
        next_state, reward, terminated, info = envn.step(action)
        next_state = np.reshape(next_state, [1, 1])
        agent.store_experience(state, action, reward, next_state, terminated)

        state = next_state
        total_reward += reward
        if terminated:
            agent.align_target_model()
            logger.info(f"Total Reward : {total_reward}")
            break

        if timestep % 10 == 0:
            bar.update(timestep / 10 + 1)

    bar.finish()

    # if (e + 1) % 10 == 0:
    logger.info("**********************************")
    logger.info("Episode: {} => Total Reward : {}".format((e+1), total_reward))
    envn.render()
    logger.info("**********************************")