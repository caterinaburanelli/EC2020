#### https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
## discrete recombination: one pont crossover (positional bias)
## random resetting mutation, to introduce new values

# decide:
# - if we want to change the pop completely after every generation or to keep
#       the best half, make this half mate and create a new pop with the best 
#       half and the offspring of the best half (problem of incest)
# - we do mutation only on the offspring or in all the population?

################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################


# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
from math import fabs,sqrt
import glob, os
import numpy

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# normalizes
def norm(x, pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# evaluation
def evaluate(x):
    return numpy.array(list(map(lambda y: simulation(env,y), x)))


def select_mating_pool(pop, fitness):

    # Selecting the parents as the first best half of the population
    parent_num = pop.shape[0]/2
    parents = numpy.empty((parent_num, pop.shape[1]))

    for parent_num in range(parent_num):
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

# the first weigths are randomly generated
sol_per_pop =  10 # how many individuals
pop = numpy.random.uniform(-1, 1, (sol_per_pop, n_vars))
pop_size = (sol_per_pop,n_vars) # the pop has sol_per_pop individuals
                                  # with num_vars genes, it's a matrix

num_generations = 2 # number of generation we implement the sequence

for generation in range(num_generations):
     # Measuring the fitness of each chromosome in the population.
     fitness = evaluate(pop)
     # Selecting the best parents in the population for mating.
     parents = select_mating_pool(pop, fitness)
     # Generating next generation using crossover.
     offspring_crossover = crossover(parents) 
     # Adding some variations to the offsrping using mutation.
     offspring_mutation = mutation(offspring_crossover)
     # Creating the new population based on the parents and offspring.
     pop[0:parents.shape[0], :] = parents
     pop[1:parents.shape[0]:, :] = offspring_mutation
     print(pop.shape)

     # Calculates mean, std and best

     best = numpy.argmax(fitness)
     mean = numpy.mean(fitness)
     std = numpy.std(fitness)

     # saves results for first pop
     file_aux  = open(experiment_name+'/results.txt','a')
     file_aux.write('\n\ngen best mean std')
     
     file_aux.write('\n'+str(generation)+' '+str(round(fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
     file_aux.close()
pop[0:parents.shape[0], :] = parents
pop[parents.shape[0]:, :] = offspring_mutation