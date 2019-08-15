import gym
import random
import numpy as np
from gym.envs.registration import register
import time
from os import system
import pickle

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
            print(f"Action Range: {self.action_low, self.action_high}")

    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)
        return action

# QAgent class with Q learning implemented to solve the problem
class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        # Total number of states
        self.state_size = env.observation_space.n
        print(f"total number of states: {self.state_size}")
        self.build_model()
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = 1.0 

    # Build Q table
    def build_model(self):
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])

    # Get action according to Q value
    def get_action(self, state):
        q_state = self.q_table[state]
        if self.epsilon > random.random():
            action = super().get_action(state)
        else:
            action = np.argmax(q_state)
        return action
        

    def train_agent(self, experience):
        state, action, next_state, reward, done = experience

        q_next = self.q_table[next_state]
        if done:
            q_next = np.zeros([self.action_size])

        q_target = reward + self.discount_rate * np.max(q_next)

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update

        if done:
            self.epsilon *= 0.994
            

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
    print(f"Observation Space: {env.observation_space} Action Space: {env.action_space}")
 
    # This is our agent
    my_agent = QAgent(env)

    # Number of episodes we want to iterate
    EPISODES = 500

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
            
            print(f"state: {state}, action: {action}")
            print(f"Episode: {episode} total reward: {total_reward} epsilon: {my_agent.epsilon}")
            env.render()
            if not done:
                _ = system("clear")

    
    pickle_out = open(f"frozenlake.pickle", "wb")
    pickle.dump(my_agent.q_table, pickle_out)
    pickle_out.close()
    

    env.close()
