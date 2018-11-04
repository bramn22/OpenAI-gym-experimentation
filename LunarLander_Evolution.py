
import gym
import numpy as np
import random
import copy

env = gym.make('LunarLander-v2')
print(env.action_space) # left, nothing, right
print(env.observation_space) # position, velocity

input_dim = 9
output_dim = 4

def select_action(obs, weights):
    obs_matrix = np.matrix(obs)
    outputs = np.hstack([np.ones([obs_matrix.shape[0], 1]), obs_matrix]) * weights
    return np.argmax(outputs)

def create_child(p1, p2):
    child = copy.deepcopy(p1)
    for i in range(input_dim):
        for j in range(output_dim):
            if random.random() > 0.5:
                child[i, j] = p2[i, j]
    if random.random() < 0.5:
        child = child + np.random.normal(0, 0.3, size=(input_dim, output_dim))
    return child

top_pop_keep = 5
pop_size = 15
random_selection_size = 5
def generate_offspring(population, rewards):
    next_generation = []
    sorted_population = [x for _,x in sorted(zip(rewards,population), reverse=True)]
    sorted_rewards = sorted(rewards, reverse=True)
    sorted_rewards_nn = [r + min(sorted_rewards) for r in sorted_rewards] # make non-negative
    normalized_rewards = [float(r) / sum(sorted_rewards_nn[top_pop_keep:]) for r in sorted_rewards_nn[top_pop_keep:]]

    next_generation.extend(sorted_population[:top_pop_keep]) # add top population
    choices = np.random.choice(np.array(range(pop_size-top_pop_keep)) + top_pop_keep, random_selection_size, p=normalized_rewards)
    next_generation.extend([sorted_population[i] for i in choices])
    for i in range(pop_size - top_pop_keep - random_selection_size):
        next_generation.append(create_child(next_generation[i], next_generation[i+1]))
    return next_generation
    #new_weights = best_weights + np.random.normal(0, 1, size=(input_dim, output_dim))


def fitness(model):
    reward = 0
    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            action = select_action(obs, model)
            obs, r, done, info = env.step(action)
            reward += r
    reward /= 10
    return reward

EPISODES = 10
population = [np.random.rand(input_dim, output_dim) for _ in range(pop_size)]
for e in range(EPISODES):
    rewards = []
    for m in population:
        rewards.append(fitness(m))
    rewards_avg = np.average(rewards)
    sorted_population = [x for _,x in sorted(zip(rewards,population), reverse=True)]

    print(e, " finished. Avg reward: ", rewards_avg, ". TOP: ", [int(np.mean(x)*10000) for x in sorted_population[:5]], ", with, ", max(rewards))

    population = generate_offspring(population, rewards)

#wait for user input
input("Press key to simulate")
for _ in range(50):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = select_action(obs, sorted_population[0])
        obs, r, done, info = env.step(action)



