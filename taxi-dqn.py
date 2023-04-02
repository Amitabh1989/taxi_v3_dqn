__doc__ = """
     The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    ### Description
    There are four designated locations in the grid world indicated by R(ed),
    G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off
    at a random square and the passenger is at a random location. The taxi
    drives to the passenger's location, picks up the passenger, drives to the
    passenger's destination (another one of the four specified locations), and
    then drops off the passenger. Once the passenger is dropped off, the episode ends.
    Map:
        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+
    ### Actions
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    ### Observations
    There are 500 discrete states since there are 25 taxi positions, 5 possible
    locations of the passenger (including the case when the passenger is in the
    taxi), and 4 destination locations.
    Note that there are 400 states that can actually be reached during an
    episode. The missing states correspond to situations in which the passenger
    is at the same location as their destination, as this typically signals the
    end of an episode. Four additional states can be observed right after a
    successful episodes, when both the passenger and the taxi are at the destination.
    This gives a total of 404 reachable discrete states.
    Each state space is represented by the tuple:
    (taxi_row, taxi_col, passenger_location, destination)
    An observation is an integer that encodes the corresponding state.
    The state tuple can then be decoded with the "decode" method.
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    ### Info
    ``step`` and ``reset()`` will return an info dictionary that contains "p" and "action_mask" containing
        the probability that the state is taken and a mask of what actions will result in a change of state to speed up training.
    As Taxi's initial state is a stochastic, the "p" key represents the probability of the
    transition however this value is currently bugged being 1.0, this will be fixed soon.
    As the steps are deterministic, "p" represents the probability of the transition which is always 1.0
    For some cases, taking an action will have no effect on the state of the agent.
    In v0.25.0, ``info["action_mask"]`` contains a np.ndarray for each of the action specifying
    if the action will change the state.
    To sample a modifying action, use ``action = env.action_space.sample(info["action_mask"])``
    Or with a Q-value based algorithm ``action = np.argmax(q_values[obs, np.where(info["action_mask"] == 1)[0]])``.
    ### Rewards
    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10  executing "pickup" and "drop-off" actions illegally.
    ### Arguments
    ```
    gym.make('Taxi-v3')
    ```
    ### Version History
    * v3: Map Correction + Cleaner Domain Description, v0.25.0 action masking added to the reset and step information
    * v2: Disallow Taxi start location = goal location, Update Taxi observations in the rollout, Update Taxi reward threshold.
    * v1: Remove (3,2) from locs, add passidx<4 check
    * v0: Initial versions release
"""

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

        logger.info(f"In ACT : state value : {state}")
        q_values = self.q_network.predict(state)
        logger.info(f"In ACT : qvalue value : {q_values}")
        logger.info(f"In ACT : qvalue value npmax: {np.argmax(q_values[0])}")
        return np.argmax(q_values[0])

    def retrain(self, batch_size):   # equal to experience_replay
        minibatch = random.sample(self.experience_replay, batch_size)

        for i, (state, action, reward, next_state, terminated) in enumerate(minibatch):
            logger.info(f">>>>>>>>>>>>>>>>> Loop {i}")
            target = self.q_network.predict(state)

            if terminated:
                target[0][action] = reward
                logger.info(f"Target Network data at terminated condition : {target}  : Action chosen : {target[0][action]}")

            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
                logger.info(f"Target Network data at target.predict : {target}  : Action chosend : {target[0][action]}")
            
            # logger.info(f"Target Network in retrain : {target}")
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
        
        if len(agent.experience_replay) > batch_size:
            agent.retrain(batch_size)

        if timestep % 10 == 0:
            bar.update(timestep / 10 + 1)

    bar.finish()

    # if (e + 1) % 10 == 0:
    logger.info("**********************************")
    logger.info("Episode: {} => Total Reward : {}".format((e+1), total_reward))
    envn.render()
    logger.info("**********************************")