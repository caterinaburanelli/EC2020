#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1
#  imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import random
import deap
import numpy as np
import glob, os

from deap import base
from deap import creator
from deap import tools
from math import fabs,sqrt


experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize the amout of neurons
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

#---------------------


creator.create("FitnessMin", base.Fitness, weights=(-30.0, -30.0))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.uniform, -1, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, n_vars)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized

# evaluation function that is used in the DEAP algorithm
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x))),

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evaluate)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selWorst)

creator.create("FitnessMin", base.Fitness, weights=(-100,))

#----------

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

def main():
    # fitnesses = np.array([])
    random.seed(126)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=60)
    pop_array = np.array(pop)

    # CXPB  is the probability with which two individuals
    #       are crossed
    # MUTPB is the probability for mutating an individual
    MUTPB = 1
    CXPB = 0.8
    
    print("Start of evolution")
    
    # Evaluates the entire population
    fitnesses = evaluate(pop_array)[0].tolist()
    for count, individual in enumerate(fitnesses):

        # Rewrites the fitness value in a way the DEAP algorithm can understand
        fitnesses[count] = (individual, )
   
   # Gives individual a fitness value
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while max(fits) < 100 and g < 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Prints fitnesses of the population (does nothing for the algorithm)
        for i in pop:
            print(i.fitness.values[0])

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Prints fitnesses of the offspring (does nothing for the algorithm
        for i in offspring:
            print(i.fitness.values[0])

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:

                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values


        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        pop_array = np.array(invalid_ind)

        fitnesses = evaluate(pop_array)[0].tolist()
        for count, individual in enumerate(fitnesses):
            fitnesses[count] = (individual, )
    
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))


        
        print("\n prints the fitness values of pop before")
        for i in pop:

            print(f"{i.fitness.values[0]}")
        print("\n")

        # Changes best individuals of population with worst individuals of the offspring

        best_gen = deap.tools.selWorst(pop, 15, fit_attr='fitness')
        worst_offspring = deap.tools.selBest(offspring, 15, fit_attr='fitness')

        for count, individual in enumerate(worst_offspring):
            index = offspring.index(individual)
            offspring[index] = best_gen[count]

        # The population is entirely replaced by the offspring (plus best of previous generations)
        pop[:] = offspring
        print(f"There are {pop} individuals in the population ")
 
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        # saves results for first pop
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\n\ngen best mean std')
        file_aux.write('\n Generation '+str(g)+' '+str(round(max(fits),6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
main()