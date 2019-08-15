import gym
import random
import numpy as np
import sys

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

env = gym.make("MountainCar-v0")
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 3000

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

for episode in range(EPISODES):
    try:
        discrete_state = get_discrete_state(env.reset())
        done = False

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])

            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            
            if episode % SHOW_EVERY == 0:
                env.render()
            

            if not done:
                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])

                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (action,)]

                # Here is our equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                # Update Q table with new Q value
                q_table[discrete_state + (action,)] = new_q

            # Simulation ended (for any reason) - if goal position is achieved -update Qvalue with reward directly
            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action, )] = 0

            discrete_state = new_discrete_state

            if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value

    except KeyboardInterrupt:
        print("closing env")
        env.close()
        sys.exit()

env.close()
