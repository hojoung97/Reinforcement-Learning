import gym

class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("action size: ", self.action_size)

    def get_action(self, state):
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action
    
env = gym.make("CartPole-v0")


cart_agent = Agent(env)
EPISODES = 200
EVERY = 20

for episode in range(EPISODES):
    done = False
    state = env.reset()
    while not done:
        action = cart_agent.get_action(state)
        state, reward, done, info = env.step(action)
        env.render()
