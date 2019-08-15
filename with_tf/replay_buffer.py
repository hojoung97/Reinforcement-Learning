import gym
import random
import numpy as np
from gym.envs.registration import register
import time
import tensorflow as tf
from collections import deque

# Agent class definition
# This is an agent that can handle both discrete and continuous action space environment
class Agent:
    def __init__(self, env):
        self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete
        if self.is_discrete:
            print("env discrete")
            self.action_size = env.action_space.n
        else:
            print("env continuous")
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action Range: {}, {}".format(self.action_low, self.action_high))

    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)
        return action

# QNAgent class with Neural network Q learning implemented to solve the problem
class QNAgent(Agent):
    ## Reduced learning rate
    def __init__(self, env, discount_rate=0.97, learning_rate=0.001):
        super().__init__(env)
        # Total number of states
        self.state_size = env.observation_space.n
        print("total number of states: {}".format(self.state_size))
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = 1.0

        self.build_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.replay_buffer = deque(maxlen=1000)

    # Build Q table
    def build_model(self):
        tf.reset_default_graph()
        self.state_in = tf.placeholder(tf.int32, shape=[None])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.target_in = tf.placeholder(tf.float32, shape=[None])
        
        self.state = tf.one_hot(self.state_in, depth=self.state_size)
        self.action = tf.one_hot(self.action_in, depth=self.action_size)
        
        self.q_state = tf.layers.dense(self.state, units=self.action_size, name="q_table")
        self.q_action = tf.reduce_sum(tf.multiply(self.q_state, self.action), axis=1)
        
        self.loss = tf.reduce_sum(tf.square(self.target_in - self.q_action))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    # Get action according to Q value
    def get_action(self, state):
        q_state = self.sess.run(self.q_state, feed_dict={self.state_in: [state]})
        if self.epsilon > random.random():
            action = super().get_action(state)
        else:
            action = np.argmax(q_state)
        return action
        

    def train_agent(self, experience, batch_size=50):
        self.replay_buffer.append(experience)
        samples = random.choices(self.replay_buffer, k=batch_size)
        state, action, next_state, reward, done = (list(col) for col in zip(experience, *samples))
        # state, action, next_state, reward, done = ([exp] for exp in experience)
        
        q_next = self.sess.run(self.q_state, feed_dict={self.state_in: next_state})
        q_next[done] = np.zeros([self.action_size])
        q_target = reward + self.discount_rate * np.max(q_next, axis=1)
        
        feed = {self.state_in: state, self.action_in: action, self.target_in: q_target}
        self.sess.run(self.optimizer, feed_dict=feed)
        
        if experience[4]:
            self.epsilon *= 0.99
            
    def __del__(self):
        self.sess.close()
            

# Main Body
if __name__ == "__main__":

    # We will modify the frozen lake environment with no slippery condition
    try:
        register(
            id='FrozenLakeNoSlip-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name' : '4x4', 'is_slippery': False},
            max_episode_steps=100,
            reward_threshold=0.78, # optimum = .8196
        )
    except:
        pass

    # This is list of the environments
    envs = ( "CartPole-v1",                         #0
             "MountainCar-v0",                      #1
             "MountainCarContinuous-v0",
             "Acrobot-v1",
             "Pendulum-v0",
             "FrozenLake-v0",
             "FrozenLakeNoSlip-v0")                 #6

    # Initialize the environment we want by choosing one from the list
    env = gym.make(envs[6])     # Change the index
    print("Observation Space: {} Action Space: {}".format(env.observation_space, env.action_space))
 
    # This is our agent
    my_agent = QNAgent(env)

    # Number of episodes we want to iterate
    EPISODES = 200

    # Run multiple episodes and let our agent play!
    total_reward = 0
    for episode in range(EPISODES):
        done = False
        state = env.reset()
        while not done:
            # Get action according to current state
            action = my_agent.get_action(state)
            # Save the conseqeunce ofchoosing the move we made
            next_state, reward, done, info = env.step(action)
            # and use the information above to train our agent
            my_agent.train_agent((state, action, next_state, reward, done))

            # Update variables
            state = next_state
            total_reward += reward
            
            print("state: {}, action: {}".format(state, action))
            print("Episode: {} total reward: {} epsilon: {}".format(episode, total_reward, my_agent.epsilon))
            env.render()

            # Print Q table
            with tf.variable_scope("q_table", reuse=True):
                weights = my_agent.sess.run(tf.get_variable("kernel"))
                print(weights)

            # sleep...
            time.sleep(0.05)

    env.close()
