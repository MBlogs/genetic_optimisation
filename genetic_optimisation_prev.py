# %% Imports and directories
import os
import sys
import random
import string
import math

def initialise_population(n, length):
  ''' Returns list of string, population size n each of length length'''
  return [initialise_individual(length) for _ in range(n)]


def initialise_individual(length):
  '''Returns random string, length chars long'''
  letters = string.ascii_uppercase + string.digits + "_"
  return ''.join(random.choice(letters) for _ in range(length))


def sort_by_fitness(population_dict):
  '''Takes individual:fitness population dictionary and returns list'''
  return [k for k, v in sorted(population_dict.items(), key=lambda item: item[1], reverse=True)]


def select_parent(population_dict, tournament_size):
  '''Takes individual:fintness dictionary.
  Returns a parent. Probability of returning chosen via tournament selection
  '''
  # Choose random subset, size tournament_size
  tournament_pool = dict(random.sample(population_dict.items(), tournament_size))
  # Order the random entries taken from the tournament pool
  ordered_pool = sort_by_fitness(tournament_pool)
  # Select best
  parent = ordered_pool[0]
  return parent


def crossover_parents(p1, p2):
  '''Crossover parent strings at random location'''
  crossover_point = random.choice(range(len(p1) - 1))
  return p1[:crossover_point] + p2[crossover_point:]


def mutate_population(population, mutation_rate):
  return [mutate_individual(i, mutation_rate) for i in population]


def mutate_individual(individual, mutation_rate):
  '''Takes individual string, returns mutated string'''
  letters = string.ascii_uppercase + string.digits + "_"
  individual_list = list(individual)
  # Mutate each character with chance mutation_rate
  for char in range(len(individual)):
    if random.random() < mutation_rate:
      individual_list[char] = random.choice(letters)
  return "".join(individual_list)


def average_population_fitness(population_dict):
  cuml_fitness = 0
  for k, v in population_dict.items():
    cuml_fitness += v
  return cuml_fitness / len(population_dict)


def produce_next_generation(population_dict, population_size, elitism_n, tournament_size, mutation_rate):
  '''Takes population dictionary with fitnesses.
  Returns new population as list
  '''
  new_population = []
  # Create new individuals in remaining slots
  for _ in range(population_size - elitism_n):
    parent1 = select_parent(population_dict, tournament_size)
    parent2 = select_parent(population_dict, tournament_size)
    new_population.append(crossover_parents(parent1, parent2))

  # Mutate the new population
  new_population = mutate_population(new_population, mutation_rate)

  # Elitism: Carry over best individuals
  sorted_population = sort_by_fitness(population_dict)
  elite_population = sorted_population[:elitism_n]

  return new_population + elite_population


def get_best_solution(population_dict):
  '''Takes population_dict, returns if any have fitness 1'''
  best_solution = ('dummy', 0)
  for k, v in population_dict.items():
    if v > best_solution[1]:
      best_solution = (k, v)
  return best_solution


def run_genetic_algorithm(population_size,
                          elitimsm_n,
                          tournament_size,
                          mutation_rate):
  # Initialise random population
  population = initialise_population(population_size, PASSWORD_LENGTH)
  population_dict = blurry_memory(population, STUDENT_NUMBER, INDEX)
  best_solution = get_best_solution(population_dict)
  average_fitness_list = [average_population_fitness(population_dict)]
  best_fitness_list = [best_solution[1]]
  N = 0

  # While fitness of best solution below 1
  while (best_solution[1] < 1 and N < 20000):
    new_population = produce_next_generation(population_dict,
                                             population_size,
                                             elitimsm_n,
                                             tournament_size,
                                             mutation_rate)
    population_dict = blurry_memory(new_population, STUDENT_NUMBER, INDEX)
    # Update population metrics
    N += 1
    best_solution = get_best_solution(population_dict)
    average_fitness = average_population_fitness(population_dict)
    average_fitness_list.append(average_fitness)
    best_fitness_list.append(best_solution[1])

    # print("Iteration: {}, Best {}, Fitness: {}, Average Fitness {}".format(N,best_solution[0],best_solution[1],average_fitness))

  return best_solution, N, average_fitness_list, best_fitness_list


# %% - - - - FIND SOLUTIONS - - - -
# Constants
PASSWORD_LENGTH = 10
STUDENT_NUMBER = 190735252
INDEX = 0

# %%
# Variables
population_size = 100  # Population size each generation
elitism_n = 2  # Number of best solutions kept
tournament_size = 5  # Number of solutions in subset each selection
mutation_rate = 0.1  # Chance of mutating character in child

INDEX = 0
solution0, N, avg, best = run_genetic_algorithm(population_size,
                                                elitism_n,
                                                tournament_size,
                                                mutation_rate)
INDEX = 1
solution1, N, avg, best = run_genetic_algorithm(population_size,
                                                elitism_n,
                                                tournament_size,
                                                mutation_rate)
print("Q1A: Password 0: {}, Password 1: {}".format(solution0[0], solution1[0]))
# %% - - - - NUMBER OF REPRODUCTIONS  - - - -
population_size = 100  # Population size each generation
elitism_n = 2  # Number of best solutions kept
tournament_size = 5  # Number of solutions in subset each selection
mutation_rate = 0.1  # Chance of mutating character in child

number_generations = []
iterations = 500

for i in range(iterations):
  solution1, N, avg, best = run_genetic_algorithm(population_size,
                                                  elitism_n,
                                                  tournament_size,
                                                  mutation_rate)
  print("Q1c Number of Reproductions Iteration:" + str(i))
  number_generations.append(N)


# %% - - - - NUMBER OF REPRODUCTIONS: PLOT RESULTS - - - -
avg_generations = sum(number_generations) / len(number_generations)
std_generations = math.sqrt(sum((xi - avg_generations) ** 2 for xi in number_generations) / len(number_generations))

number_reproductions = [100 * g for g in number_generations]
avg_reproductions = sum(number_reproductions) / len(number_reproductions)
std_reproductions = math.sqrt(
  sum((xi - avg_reproductions) ** 2 for xi in number_reproductions) / len(number_reproductions))

print("Avg generations: {} , Std generations: {}".format(avg_generations, std_generations))
print("Avg reproductions: {} , Std reproductions: {}".format(avg_reproductions, std_reproductions))

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.hist(number_generations, bins=max(number_generations) - min(number_generations))
ax.set_ylabel('Number of trials')
ax.set_xlabel('Number of generations until solution')
plt.show()

# %% - - - - IMPACT OF VARYING PARAMETER POPULATION SIZE - - - -
# Will change population_size each time
trials_per_population = 100
population_sizes = [20, 40, 60, 80, 100, 120, 140]
elitism_n = 2  # Number of best solutions kept
tournament_size = 5  # Number of solutions in subset each selection
mutation_rate = 0.1  # Chance of mutating character in child
# Store generation
average_fitness_dict = {p: [[] for _ in range(50)] for p in population_sizes}

for population_size in population_sizes:
  for trial in range(trials_per_population):
    solution, N, avg, best = run_genetic_algorithm(population_size,
                                                   elitism_n,
                                                   tournament_size,
                                                   mutation_rate)
    for i in range(len(avg)):
      if i < len(average_fitness_dict[population_size]):
        average_fitness_dict[population_size][i].append(avg[i])

  print("Q1d: Done population size: {}".format(population_size))

# %% - - - - IMPACT OF VARYING PARAMETER: PLOT RESULTS - - - -

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
overall_avgs = {}
for p in population_sizes:
  overall_avgs[p] = [sum(gen) / len(gen) for gen in average_fitness_dict[p] if len(gen) > 0]
  ax.plot(overall_avgs[p], label=p)
ax.set_ylabel('Average Fitness of generation')
ax.set_xlabel('Generation')
ax.legend(loc="lower right", title="Population sizes")
ax.set_xlim(0, 50)
ax.set_ylim(0.5, 0.9)
plt.show()

# %% - - - - IMPACT OF VARYING PARAMETER MUTATION RATE - - - -
trials_per_population = 100
population_size = 100
mutation_rates = [0.05, 0.1, 0.15, 0.2, 0.25]
elitism_n = 2  # Number of best solutions kept
tournament_size = 5  # Number of solutions in subset each selection
mutation_dict = {p: [] for p in mutation_rates}

for mutation_rate in mutation_rates:
  for trial in range(trials_per_population):
    solution, N, avg, best = run_genetic_algorithm(population_size,
                                                   elitism_n,
                                                   tournament_size,
                                                   mutation_rate)
    mutation_dict[mutation_rate].append(N)
  print("Q1d: Done Mutation Rate: {}".format(mutation_rate))
# %% - - - - IMPACT OF VARYING PARAMETER:MUTATION RATE REPORT RESULTS - - - -
overall_avg = {}
overall_std = {}
for m in mutation_rates:
  overall_avg[m] = sum(mutation_dict[m]) / len(mutation_dict[m])
  overall_std[m] = math.sqrt(sum((xi - overall_avg[m]) ** 2 for xi in mutation_dict[m]) / len(mutation_dict[m]))
  print("Mutation Rate:{} Average:{} , Std:{} ".format(m, overall_avg[m], overall_std[m]))