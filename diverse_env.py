import gym
import random
import numpy as np

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

# Main Body
if __name__ == "__main__":

    # This is list of the environments
    envs = ( "CartPole-v1",
             "MountainCar-v0",
             "MountainCarContinuous-v0",
             "Acrobot-v1",
             "Pendulum-v0",
             "FrozenLake-v0")

    # Initialize the environment we want
    env = gym.make(envs[5])
    print(f"Observation Space: {env.observation_space} Action Space: {env.action_space}")

    # This is our agent
    cart_agent = Agent(env)

    # Number of episodes we want to iterate
    EPISODES = 5

    # Run multiple episodes and let our agent play!
    for _ in range(EPISODES):
        done = False
        state = env.reset()
        while not done:
            action = cart_agent.get_action(state)
            state, reward, done, info = env.step(action)
            env.render()

    env.close()
