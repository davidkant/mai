import random
import functools
import matplotlib.pyplot as plt
import numpy as np
import copy

# default functions ----------------------------------------------------------------------

def random_individual():
    """Generate a random individual phenotype"""

    return [random.randrange(0,2) for i in range(10)]

def to_phenotype(genotype):
    """Trivial mapping genotype -> phenotype (for now...)"""

    return genotype

def fitness_func(individual):
    """Evaluate the fitness of an individual using hamming
    distance to [1,1, ... , 1]. returns value within [0,1]
    """

    # ideal vector
    target = [1] * len(individual)

    # hamming distance to ideal vector
    distance = sum([abs(x - y) for (x,y) in zip(individual, target)]) / float(len(target))

    # invert for fitness
    return 1 - distance

def to_weight(fitness, m=100, b=1):
    """Convert from fitness score to probability weighting"""

    return int(round(fitness*m + b))

def reproduce(parent1, parent2):
    """generate offspring using random crossover"""

    # random crossover point
    crossover = random.randrange(0, len(parent1))

    # construct children
    child1 = parent1[0:crossover] + parent2[crossover:]
    child2 = parent2[0:crossover] + parent1[crossover:]

    # return children
    return child1, child2

def mutate(genotype, mutation_prob=0.01, inbreeding_prob=0.5, verbose=True):
    """Mutate!"""

    # do we mutate?
    if random.random() <= mutation_prob:

        # print it
        if verbose: print('-> muuuuutating individual {0}'.format(genotype))

        # select a random chromosome
        gene_index = random.randrange(len(genotype))

        # flip its value
        genotype[gene_index] = 1 - genotype[gene_index]

    return genotype

# genetic algorithm  ---------------------------------------------------------------------

class GeneticAlgorithm:
    """A very simple Genetic Algorithm."""

    def __init__(self):

        # initialize default functions
        self.random_individual = random_individual
        self.to_phenotype = to_phenotype
        self.fitness_func = fitness_func
        self.to_weight = to_weight
        self.reproduce = reproduce
        self.mutate = mutate

    def initialize_population(self, population_size=10):
        """Initialize the population."""

        # store population size
        self.population_size = population_size

        # initialize individuals
        self.population = [self.random_individual() for i in range(population_size)]
        self.generations = [copy(self.population)]

        # initialize fitness to 0 for all
        self.fitness = [[0, individual] for individual in self.population]

    def evolve(self, iters=10, population_size=100, init_pop=True, mutation_prob=0.01):
        """Run the GA."""

        # initialize the population
        if init_pop or self.population == None:
            self.population = [self.random_individual() for i in range(population_size)]
            self.generations = [copy(self.population)]

        # loop iters times
        for i in range(iters):

            # evaluate fitness over the entire population
            self.fitness = [(self.fitness_func(self.to_phenotype(individual)), individual)
                       for individual in self.population]

            # construct mating pool of probabilities weighted by fitness score
            mating_pool = functools.reduce(lambda x,y: x+y, [[individual]*self.to_weight(score)
                                                   for (score,individual) in self.fitness])

            # select population_size/2 pairs of parents from the mating pool
            parents = [(random.choice(mating_pool), random.choice(mating_pool))
                       for i in range(int(population_size/2))]

            # generate new offspring from parents
            offspring = functools.reduce(lambda x,y: x+y, [self.reproduce(parent1, parent2)
                                                 for (parent1,parent2) in parents])

            # mutate
            map(lambda x: self.mutate(x, mutation_prob=mutation_prob), offspring)

            # update the population
            self.population = offspring
            self.generations += [copy(self.population)]

        return self.population

    def evolve_once(self, mutation_prob=0.01):
        """Evolve one generation using fitness scores in self.fitness."""

        # construct mating pool of probabilities weighted by fitness score
        mating_pool = functools.reduce(lambda x,y: x+y, [[individual]*self.to_weight(score)
                                               for (score,individual) in self.fitness])

        # select population_size/2 pairs of parents from the mating pool
        parents = [(random.choice(mating_pool), random.choice(mating_pool))
                   for i in range(int(self.population_size/2))]

        # generate new offspring from parents
        offspring = functools.reduce(lambda x,y: x+y, [self.reproduce(parent1, parent2)
                                             for (parent1,parent2) in parents])

        # mutate
        offspring = [self.mutate(x, mutation_prob=mutation_prob) for x in offspring]

        # update the population
        self.population = offspring
        self.generations = [copy(self.population)]

        # update individuals in the fitness
        self.fitness = [[0, individual] for individual in self.population]

    def set_fitness(self, score, individual=0):
        """Set individual fitness score."""

        # update fitness score
        self.fitness[individual][0] = score


def plot_genotype(genotype):
    """Plot genotype as matrix."""
    plt.figure(figsize=(3,0.5))
    plt.imshow(np.atleast_2d(np.array(genotype)), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def test_GA():
    ga = GeneticAlgorithm()
    ga.evolve(10, init_pop=True, mutation_prob=0.01)
    return ga
