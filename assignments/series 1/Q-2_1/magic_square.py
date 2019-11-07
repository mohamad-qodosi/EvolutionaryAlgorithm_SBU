import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

L = 6  # square is L * L
fitness_evaluation_counts = 0
magic_number = (L ** 3 - L) / 2

def fitness(genotype):
    global fitness_evaluation_counts, magic_number
    fitness_evaluation_counts += 1

    arrayGenom = np.array(genotype).reshape(L, L)
    error = np.sum(np.abs(np.sum(arrayGenom, axis=0) - magic_number) ** 2)
    error += np.sum(np.abs(np.sum(arrayGenom, axis=1) - magic_number) ** 2)
    error += np.sum(np.abs(np.trace(arrayGenom) - magic_number) ** 2)
    error += np.sum(np.abs(np.trace(arrayGenom[:, ::-1]) - magic_number) ** 2)

    return magic_number * (2 * L + 2) - error


def print_genotype(genotype):
    square = np.array(genotype).reshape(L, L)
    print(square)
    print(np.sum(square, axis=0))
    print(np.sum(square, axis=1))
    print(np.trace(square))
    print(np.trace(square[:, ::-1]))
    print(square.reshape(-1).sort())

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
    for i in range(50):
        for i in range(2):
            index = roletroll(cumulative_chance)
            parents.append(population[index])

    random.shuffle(parents)
    parents = [(parents[i], parents[i + 1]) for i in range(0, 100, 2)]
    return parents

def recombination(parents):
    cut_off_start = np.random.randint(0, L * L)
    cut_off_end = np.random.randint(0, L * L)
    if cut_off_end < cut_off_start:
        cut_off_end, cut_off_start = cut_off_start, cut_off_end

    parent1 = parents[0]['chromosome']
    parent2 = parents[1]['chromosome']
    offspring1 = parent1[cut_off_start:cut_off_end]
    offspring2 = parent2[cut_off_start:cut_off_end]

    for i in range(L * L):
        if parent2[(i + cut_off_end) % (L * L)] not in offspring1:
            offspring1.append(parent2[(i + cut_off_end) % (L * L)])

        if parent1[(i + cut_off_end) % (L * L)] not in offspring2:
            offspring2.append(parent1[(i + cut_off_end) % (L * L)])

    return offspring1, offspring2


def mutation(offspring):
    chance = np.random.rand()
    if chance < 0.8:
        i, j = random.randint(0, L * L - 1), random.randint(0, L * L - 1)
        print(i, j)
        offspring[i], offspring[j] = offspring[j], offspring[i]

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

population = [list(np.random.permutation(L * L) + 1) for i in range(100)]
population = [{'chromosome': genotype, 'fitness': fitness(genotype)} for genotype in population]
population = sorted(population, key=lambda x: x['fitness'])

best_fitness = 0
all_fitness = []

while best_fitness != magic_number * (2 * L + 2) and fitness_evaluation_counts < 10 * 1000:
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
