#### https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
## discrete recombination: one pont crossover (positional bias)
## random resetting mutation, to introduce new values

# decide:
# - if we want to change the pop completely after every generation or to keep
#       the best half, make this half mate and create a new pop with the best 
#       half and the offspring of the best half (problem of incest)
# - we do mutation only on the offspring or in all the population?

import ga
import numpy

num_vars = # number of genes
# the first weigths are randomly generated
weigths = numpy.random.uniform(low=-1.0, high=1.0, size=num_vars)
sol_per_pop =  100 # how many individuals
pop_size = (sol_per_pop,num_vars) # the pop has sol_per_pop individuals
                                  # with num_vars genes, it's a matrix

# random population
new_population = numpy.random.uniform(low=-1.0, high=1.0, size=pop_size)

num_generations = 10 # number of generation we implement the sequence

for generation in range(num_generations):
     # Measuring the fitness of each chromosome in the population.
     fitness = ga.cal_pop_fitness(equation_inputs, new_population)
     # Selecting the best parents in the population for mating.
     parents = ga.select_mating_pool(new_population, fitness)
     # Generating next generation using crossover.
     offspring_crossover = ga.crossover(parents) 
     # Adding some variations to the offsrping using mutation.
     offspring_mutation = ga.mutation(offspring_crossover)
     # Creating the new population based on the parents and offspring.
     new_population[0:parents.shape[0], :] = parents
     new_population[parents.shape[0]:, :] = offspring_mutation

# fitness function
def cal_pop_fitness(weigths, pop):
     # Calculating the fitness value of each solution in the current population.
     # The fitness function calculates the sum of products between each input and its corresponding weight.
     fitness = numpy.sum(pop*equation_inputs, axis=1)
     return fitness

def select_mating_pool(pop, fitness):

    # Selecting the parents as the first best half of the population
    parent_num = pop.shape[0]/2
    parents = numpy.empty((parent_num, pop.shape[1]))

    for parent_num in range(num_parents):
        # find the index for the best half
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        # once the parent is chosen, its fitness value is not important anymore
        # so I set it as a bad number so teh parent will not be chosen again
        # ====> must decide what value
        fitness[max_fitness_idx] = -99999999999

    return parents

def crossover(parents):
    offspring_size = (parents.shape[0]/2, parents.shape[1])
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents
    # a random number between 1 and l-1, with l=n_vars
    crossover_point = numpy.random.randint(1, n_vars, size=1)
 
    for k in range(offspring_size[0]):
         # Index of the parents: the mating happens with p[0]-p[1], p[1]-p[2],
         # p[2]-p[3]... untill there is no more offspring needed         
         # the last parent will give birth to only one offspring
         parent1_idx = k%parents.shape[0]
         parent2_idx = (k+1)%parents.shape[0]
         # The new offspring will have its first half of its genes taken from the first parent.
         offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
         # The new offspring will have its second half of its genes taken from the second parent.
         offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
     return offspring

def mutation(offspring_crossover):

    # Mutation substitutes a single gene (weight) in each offspring randomly.

    for idx in range(offspring_crossover.shape[0]):
        # Choosing randomly which gene to choose
        gene_tobe = numpy.random.randint(0, n_vars+1, size=1)
        # the gene_tobe of the idx-th offspring will be changed with another random weight
        # uniform mutation
        offspring_crossover[idx, gene_tobe] = numpy.random.uniform(low=-1.0, high=1.0, size=1)
    
    return offspring_crossover

new_population[0:parents.shape[0], :] = parents
new_population[parents.shape[0]:, :] = offspring_mutation