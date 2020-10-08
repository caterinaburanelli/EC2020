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
import csv
import time
import statistics
import random
import deap
import numpy as np
import glob, os

from deap import base
from deap import creator
from deap import tools
from math import fabs,sqrt

def main1(seed, game, algorithm, group):
    experiment_name = 'dummy_demo'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize the amout of neurons
    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[7,8],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    multiplemode="yes",
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
    toolbox.register("select", tools.selTournament)

    creator.create("FitnessMin", base.Fitness, weights=(-100,))

    #----------


    def simulation(env,x):
        f,p,e,t = env.play(pcont=x)
        return f, p

    # a = f'/best_game_{game}Tournement.txt'
    # # Load specialist controller
    # sol = np.loadtxt('dummy_demo' + a)
    # print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY \n')
    # evaluate([sol])

    # runs simulation
    def main(seed, game, group):
        file_aux  = open(str(algorithm)+'_group_'+str(group)+'.txt','a')
        file_aux.write(f'\ngame {game} \n')
        file_aux.write('gen, best, mean, std, median, q1, q3, life')
        file_aux.close()

        # fitnesses = np.array([])
        random.seed(seed)

        # create an initial population of 30 individuals (where
        # each individual is a list of integers)
        pop = toolbox.population(n=50)
        pop_array = np.array(pop)

        # CXPB  is the probability with which two individuals
        #       are crossed
        # MUTPB is the probability for mutating an individual
        CXPB = 0.8
        
        print("Start of evolution")
        
        # Evaluates the entire population
        
        values = evaluate(pop_array)
        values = values[0].tolist()
        fitnesses = []
        lifes = []
        for value in values:
            fitnesses.append(value[0])
            lifes.append(value[1])
        for count, individual in enumerate(fitnesses):

            # Rewrites the fitness value in a way the DEAP algorithm can understand
            fitnesses[count] = (-individual, )
    
        # Gives individual a fitness value
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of 
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0
        g_end = 1

        # Saves first generation
        length = len(pop)
        mean = sum(fits) / length * -1
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - abs(mean)**2)**0.5
        q1 = np.percentile(fits, 25) * -1
        median = np.percentile(fits, 50) * -1 
        q3 = np.percentile(fits, 75) * -1
        max_life = max(lifes) 
        file_aux  = open(str(algorithm)+'_group_'+ str(group)+ '.txt','a')
        file_aux.write(f'\n{str(g)}, {str(round(min(fits)*-1,6))}, {str(round(mean,6))}, {str(round(std,6))}, {str(round(median,6))}, {str(round(q1,6))}, {str(round(q3,6))}, {str(round(max_life,6))}')
        file_aux.close()

        # Begin the evolution
        while max(fits) < 100 and g < g_end:
            # A new generation
            g = g + 1
            print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop), 6)
            
            for i in offspring:
                print(i.fitness.values[0])
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Prints fitnesses of the offspring (does nothing for the algorithm
            # for i in offspring:
            #     print(i.fitness.values[0])

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
                if random.random() < (0.5):
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            pop_array = np.array(invalid_ind)

            values = evaluate(pop_array)
            values = values[0].tolist()
            fitnesses = []
            for value in values:
                fitnesses.append(value[0])
                lifes.append(value[1])

            for count, individual in enumerate(fitnesses):
                fitnesses[count] = (-individual, )
        
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            print("  Evaluated %i individuals" % len(invalid_ind))
            # Changes best individuals of population with worst individuals of the offspring
            amount_swithed_individuals = int(len(pop)/10)
            worst_offspring = deap.tools.selWorst(offspring, amount_swithed_individuals,fit_attr='fitness')
            best_gen = deap.tools.selBest(pop, amount_swithed_individuals, fit_attr='fitness')

            for count, individual in enumerate(worst_offspring):
                index = offspring.index(individual)
                offspring[index] = best_gen[count]

            # The population is entirely replaced by the offspring (plus best of previous generations)
            pop[:] = offspring
            print(f"There are {len(pop)} individuals in the population ")
    
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length * -1
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            q1 = np.percentile(fits, 25) * -1
            median = np.percentile(fits, 50) * -1 
            q3 = np.percentile(fits, 75) * -1
            max_life = max(lifes) 
            
            print("  Min %s" % max(fits))
            print("  Max %s" % min(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

            # saves results for first pop
            file_aux  = open(str(algorithm)+'_group_'+str(group)+'.txt','a')
            file_aux.write(f'\n{str(g)}, {str(round(min(fits) *-1,6))}, {str(round(mean,6))}, {str(round(std,6))}, {str(round(median,6))}, {str(round(q1,6))}, {str(round(q3,6))}, {str(round(max_life,6))}')
            file_aux.close()
            best_ind = tools.selBest(pop, 1)[0]
            print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            np.savetxt(experiment_name+'/Algorithm_'+ algorithm_number + '_group_'+group+'/individuals_game_'+str(game)+'/game_'+str(game)+'_gen_'+str(g)+'_group_'+str(group)+'_Tournement.txt',best_ind)
        print("-- End of (successful) evolution --")
        

    main(seed, game, group)

algorithm = 'Tournement'
group = "1"
algorithm_number = "1"
for game in range(10):
    seed = random.randint(1, 126)
    main1(seed, game ,algorithm, group)

