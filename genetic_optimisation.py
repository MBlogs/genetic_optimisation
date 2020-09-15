import random


def create_child(population, fitnesses, genes, tournament_size, mutation_rate):
  # Select parents
  parent1 = select(population, fitnesses, tournament_size)
  parent2 = select(population, fitnesses, tournament_size)
  # Crossover (breed) parents to create new individual
  child = crossover(parent1, parent2)
  # Mutate child (possibly)
  child = mutate(child, mutation_rate, genes)
  return child


def select(population, fitnesses, tournament_size):
  # Choose an individual in population by winner of a tournament
  population_indexes = random.sample(range(len(population)),tournament_size)
  tournament_population = [population[t] for t in population_indexes]
  tournament_fitnesses = [fitnesses[t] for t in population_indexes]
  tournament_winner = tournament_fitnesses.index(max(tournament_fitnesses))
  parent = population[population_indexes[tournament_winner]]
  return parent.copy()


def crossover(parent1,parent2):
  # Returns results of single point crossover of the parents
  crossover_point = random.choice(range(len(parent1)))
  child = parent1.copy()
  child[crossover_point:] = parent2[crossover_point:]
  return child


def mutate(child, mutation_rate, genes):
  for gene in range(len(child)):
    if random.random() < mutation_rate:
      child[gene] = random.choice(genes)
  return child


def population_fitnesses(population):
  # Takes list of individuals and returns population and fitness lists (sorted by fitness)
  fitnesses = [fitness(i) for i in population]
  fitnesses, population = zip(*sorted(zip(fitnesses, population),reverse=True))
  return list(population), list(fitnesses)

def _fitness(i):
  return sum(i)

def fitness(i):
  # Takes individual, returns fitness float. Needs to be customised according to use
  ascii_diff = 0
  for x in range(len("pass")):
    ascii_diff -= abs(ord("pass"[x]) - ord(''.join(i)[x]))
  return ascii_diff


def optimise(iterations, genes, genome_length, population_size, elites, tournament_size, mutation_rate):
  # Initialisation and order by fitnesses
  population = [random.sample(genes, genome_length) for _ in range(population_size)]
  population, fitnesses = population_fitnesses(population)

  # Iterate through generations of populations
  for generation in range(iterations):
    new_population = [p.copy() for p in population]

    # Keep the best elites, replace rest of population with new children
    for p in range(elites, population_size):
      new_population[p] = create_child(population, fitnesses, genes, tournament_size, mutation_rate)

    population, fitnesses = population_fitnesses(new_population)
    print(f"Best individual {population[0]} with fitness {fitnesses[0]}")

  return population[0], fitnesses[0]

if __name__ == "__main__":
  iterations = 100
  genes = "abcdefghijklmnopqrstuvwxyz"
  genome_length = 4
  #genes = [1,2,3,4,5,6,7,8,9,0]
  #genome_length = 4
  population_size = 50
  elites = 2
  tournament_size = 4
  mutation_rate = 0.05
  best, best_fitness = optimise(iterations, genes, genome_length, population_size, elites, tournament_size, mutation_rate)
  # population = [random.sample(genes, genome_length) for _ in range(population_size)]
  # population, fitnesses = population_fitnesses(population)
  # print(f"Population: {population}")
  # print(f"Fitnesses: {fitnesses}")
  # parent1 = select(population,fitnesses,tournament_size)
  # parent2 = select(population, fitnesses, tournament_size)
  # child = crossover(parent1,parent2)

