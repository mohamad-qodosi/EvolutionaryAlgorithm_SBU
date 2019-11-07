import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fitness_evaluation_counts = 0

def fitness(genotype):
    global fitness_evaluation_counts

    fitness_evaluation_counts += 1

    z = 2 * genotype[0] + genotype[1] + 3 * genotype[2]
    return 157 - abs(23 - z)

def print_genotype(genotype):
    print('x =', genotype[0], ', y =', genotype[1], ', z =', genotype[2])

def roletroll(cumulative_chance):
    p = np.random.random()
    for i in range(len(cumulative_chance)):
        if cumulative_chance[i] > p:
            return i

def select_parents(population):
    population_fitness = np.array([parent['fitness'] for parent in population])
    sum_fitness = np.sum(population_fitness)
    population_chance = population_fitness / sum_fitness
    cumulative_chance = np.cumsum(population_chance)

    parents = []
    for i in range(3):
        for i in range(2):
            index = roletroll(cumulative_chance)
            parents.append(population[index])

    parents = [(parents[i], parents[i + 1]) for i in range(0, 6, 2)]
    return parents

def recombination(parents):
    cutoff = np.random.randint(0, 3)
    parent1 = parents[0]['chromosome']
    parent2 = parents[1]['chromosome']
    offspring1 = parent1[:cutoff] + parent2[cutoff:]
    offspring2 = parent2[:cutoff] + parent1[cutoff:]
    return offspring1, offspring2


def mutation(offspring):
    for i in range(3):
        chance = np.random.rand()
        if chance < 0.1:
            offspring[i] = np.random.randint(0, 31)

    return offspring


def generate_offsprings(parents_list):
    offsprings = []
    for parents in parents_list:
        offspring1, offspring2 = recombination(parents)
        offspring1 = mutation(offspring1)
        offspring2 = mutation(offspring2)

        offsprings = offsprings + [{'chromosome': offspring1, 'fitness': fitness(offspring1)},
                                   {'chromosome': offspring2, 'fitness': fitness(offspring2)}]
    return offsprings


population = np.random.randint(0, 31, size=(6, 3)).tolist()
population = [{'chromosome': genotype, 'fitness':fitness(genotype)} for genotype in population]
population = sorted(population, key=lambda x: x['fitness'])

best_fitness = 0
all_fitness = []

while best_fitness != 157 and fitness_evaluation_counts < 10 * 1000:
    parents = select_parents(population)
    offsprings = generate_offsprings(parents)

    population = offsprings

    population = sorted(population, key=lambda x: x['fitness'])
    best_fitness = population[-1]['fitness']

    all_fitness.append(list(map(lambda x: x['fitness'], population)))

best_solution = population[-1]['chromosome']

all_fitness = np.array(all_fitness)
plt.plot(all_fitness.mean(axis=1), 'ro-', label='mean fitness')
plt.plot(all_fitness.max(axis=1), 'bo--', label='max fitness')
plt.legend()

print_genotype(best_solution)

plt.show()
