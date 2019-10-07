import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fittness_evaluation_counts = 0

def fittness(genotype):
    global fittness_evaluation_counts

    fittness_evaluation_counts += 1
    checks = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(genotype[i] - genotype[j]) == j - i:
                checks += 1

    return 1 / (1 + checks)

def print_genotype(genotype):
    print('-' * 33)
    for i in range(8):
        for j in range(8):
            if genotype[j] == i:
                print('| * ', end='')
            else:
                print('|   ', end='')
        print('|')
        print('-' * 33)

def select_parents(population):
    random_candidates = random.sample(population, 5)
    parents = sorted(random_candidates, key= lambda x: x[1])[-2:]
    return parents

def recombination(parents):
    cut_off = np.random.randint(0, 8)

    parent1 = parents[0][0]
    parent2 = parents[1][0]
    offspring1 = parent1[:cut_off]
    offspring2 = parent2[:cut_off]

    for i in range(8):
        if parent2[(i + cut_off) % 8] not in offspring1:
            offspring1.append(parent2[(i + cut_off) % 8])

        if parent1[(i + cut_off) % 8] not in offspring2:
            offspring2.append(parent1[(i + cut_off) % 8])

    return offspring1, offspring2


def mutation(offspring):
    chance = np.random.rand()
    if chance < 0.8:
        i, j = random.randint(0, 7), random.randint(0, 7)
        offspring[i], offspring[j] = offspring[j], offspring[i]

    return offspring


def generate_offsprings(parents):
    offspring1, offspring2 = recombination(parents)
    offspring1 = mutation(offspring1)
    offspring2 = mutation(offspring2)

    offsprings = [(offspring1, fittness(offspring1)), (offspring2, fittness(offspring2))]
    return offsprings


population = [list(np.random.permutation(8)) for i in range(100)]
population = [(genotype, fittness(genotype)) for genotype in population]
population = sorted(population, key=lambda x: x[1])

best_fittness = 0
all_fittness = []

while best_fittness != 1 and fittness_evaluation_counts < 10 * 1000:  # 1 means zero checks
    parents = select_parents(population)
    offsprings = generate_offsprings(parents)

    population[:2] = offsprings

    population = sorted(population, key=lambda x: x[1])
    best_fittness = population[-1][1]

    all_fittness.append(list(map(lambda x: x[1], population)))

best_solution = population[-1][0]

all_fittness = np.array(all_fittness)
plt.subplot(1, 2, 1)
plt.plot(all_fittness.mean(axis=1), 'ro-', label='mean fitness')
plt.plot(all_fittness.max(axis=1), 'bo--', label='max fitness')
plt.legend()

plt.subplot(1, 2, 2)

plt.vlines(0, 0, 8)
plt.hlines(0, 0, 8)
plt.vlines(8, 0, 8)
plt.hlines(8, 0, 8)

for i in range(8):
    for j in range(i % 2, 8, 2):
        rect = patches.Rectangle((i, j), 1, 1, linewidth=1, facecolor='g')
        plt.gca().add_patch(rect)

for i in range(8):
    plt.plot(i + 0.5, best_solution[i] + 0.5, 'kx', markersize=10)

plt.show()