import random
import matplotlib.pyplot as plt # Optional

def select(population, tournament_size):
  # Choose an individual in population through winner of a tournament
  assert tournament_size <= len(population)
  tournament_winner_index = min(random.sample(range(len(population)), tournament_size))
  parent = population[tournament_winner_index]
  return parent.copy()


def create_child(population, genes, tournament_size, mutation_rate):
  # Breed two parents, returning (possibly mutated) offspring
  parent1 = select(population, tournament_size)
  parent2 = select(population, tournament_size)
  child = crossover(parent1, parent2)
  child = mutate(child, mutation_rate, genes)
  return child


def crossover(parent1,parent2):
  # Returns results of single point crossover of the parents
  crossover_point = random.choice(range(len(parent1)))
  child = parent1.copy()
  child[crossover_point:] = parent2[crossover_point:]
  return child


def mutate(child, mutation_rate, genes):
  # Choose to replace a gene with a random different one
  for gene in range(len(child)):
    if random.random() < mutation_rate:
      child[gene] = random.choice(genes)
  return child


def population_fitnesses(population, fitness_func):
  # Takes list of individuals and returns population (sorted by fitness) and fitness lists
  fitnesses = [fitness_func(i) for i in population]
  fitnesses, population = zip(*sorted(zip(fitnesses, population), reverse=True))
  return list(population), list(fitnesses)


def optimise(iterations, genes, genome_length, fitness_func
             , population_size, elites_size, tournament_size, mutation_rate, max_fitness=None):
  # Initialisation and order by fitnesses
  population = [random.choices(genes, k=genome_length) for _ in range(population_size)]
  population, fitnesses = population_fitnesses(population, fitness_func)

  best_fitnesses = []

  # Iterate through generations of populations
  for generation in range(iterations):
    new_population = [p.copy() for p in population]

    # Keep the best elites, replace rest of population with new children
    for p in range(elites_size, population_size):
      new_population[p] = create_child(population, genes, tournament_size, mutation_rate)

    # Compute fitnesses and sort population by them
    population, fitnesses = population_fitnesses(new_population, fitness_func)
    print(f"Gen: {generation+1}, Best: {population[0]} with Fitness: {fitnesses[0]}")
    best_fitnesses.append(fitnesses[0])

    # Check if we've passed the max fitness stopping condition
    if max_fitness is not None and fitnesses[0] >= max_fitness:
      break

  return population[0], fitnesses[0], best_fitnesses


if __name__ == "__main__":
  # Example of using function to solve a password
  iterations = 1000
  population_size = 200
  elites_size = 5
  tournament_size = 10
  mutation_rate = 0.1

  genes = "abcdefghijklmnopqrstuvwxyz123456789"
  genome_length = len("password123")
  fitness_func = lambda x: sum([-abs(ord("password123"[i]) - ord(''.join(x)[i])) for i in range(len(x))])

  best, best_fitness, best_fitnesses = optimise(iterations, genes, genome_length, fitness_func
                                , population_size, elites_size, tournament_size, mutation_rate, 0)
  print(f"Best: {best} with Fitness: {best_fitness}")

  # Optional: Graph of fitness improvement over time
  fig,ax = plt.subplots()
  ax.plot(best_fitnesses)
  ax.set_ylabel('Fitness')
  ax.set_xlabel('Generation')
  ax.set_xlim(0, len(best_fitnesses))
  ax.set_ylim(best_fitnesses[0], 0)
  plt.show()

