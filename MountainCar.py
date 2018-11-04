import gym
import numpy as np

env = gym.make('MountainCar-v0')
print(env.action_space) # left, nothing, right
print(env.observation_space) # position, velocity

input_dim = 2
output_dim = 3

best_weights = np.random.rand(input_dim, output_dim)
best_steps = 200

def select_action(obs, weights):
    outputs = np.matrix(obs) * weights
    return np.argmax(outputs)

EPISODES = 1000
for e in range(EPISODES):
    #new_weights = np.random.rand(input_dim, output_dim)
    new_weights = best_weights + np.random.normal(0, 1, size=(input_dim, output_dim))
    steps = 0
    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            action = select_action(obs, new_weights)
            obs, r, done, info = env.step(action)
            steps += 1
    steps /= 10
    if steps < best_steps:
        best_weights = new_weights
        best_steps = steps
    print(e, " finished ", steps, " --- best_steps: ", best_steps)

for _ in range(10):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = select_action(obs, best_weights)
        obs, r, done, info = env.step(action)
