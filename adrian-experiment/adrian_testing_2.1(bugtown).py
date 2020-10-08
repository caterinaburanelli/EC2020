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
import matplotlib.pyplot as plt
import pandas as pd

from deap import base
from deap import creator
from deap import tools
from math import fabs,sqrt
import datetime

begin_game = datetime.datetime.now()
df = pd.DataFrame(index=[1,2,3,4,5,6,7,8,9,10,11,12],columns=["Min","Max","average","SD"])

def main1(game, enemy, algorithm):
    # Setting up the game
    experiment_name = 'adrian-testing'+"-algorithm-"+ str(algorithm)
    print(experiment_name + " game: " + str(game) + " " + "enemy: " + str(enemy))
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize the amout of neurons
    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,    # default environment fitness is assumed for experiment
                      speed="fastest")
    # default environment fitness is assumed for experiment
    env.state_to_log()  # checks environment state

    #Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
    ini = time.time()  # sets time marker

    # genetic algorithm params

    run_mode = 'train'  # train or test

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

 #------------  Setting up the GA  ---------------
    # There are two main areas where change is possible a) and b) (Also new things could be added e.g. Doomsday..,)
    # a) GA Constants and parameters
    genome_lenght = n_vars #100 #lenght of the bit string to be optimized -> bit lenght is actually n_vars
    pop_size = 50
    p_crossover=0.8
    p_mutation=0.2
    mutation_scaler = genome_lenght # The bitflip function iterates over every single value in an individuals genome, with a probability indpb it decides wether to flip or not. This value is independent from the mutation probability which decides IF a given individual in the population will be selected for mutation.

    max_generations= 15  # stopping condition
    tournament_size = 5 # tournament size
    seed = 50 #random.randint(1, 126)
    random.seed(seed)

    # For optimizing continuous functions
    bound_low = -1
    bound_up = 1
    crowding_factor = 20


    # Defining a tool to create single gene
    toolbox=base.Toolbox()
    # toolbox.register("ZeroOrOne", random.randint, -1, 1)
    toolbox.register("ZeroOrOne", random.uniform, -1, 1) # Each gene is a float between -1 and 1

    # Defining the fitness
    # creator.create("FitnessMin", base.Fitness, weights=(-30.0, -30.0))
    # creator.create("FitnessMin", base.Fitness, weights=(-100,))
    creator.create("FitnessMin", base.Fitness, weights=(100,))

    # Defining an individual creatorfre
    creator.create("Individual", list, fitness=creator.FitnessMin) # An individual will be stored in a list format with fitness evaluated at "FitnessMin"
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                          toolbox.ZeroOrOne,genome_lenght) # An individual consist of a list of n_var attributes (genes) populated by zeroorone

    # Defining the population cretor
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    # Defining the fitness function
    def evaluate(x):
        return np.array(list(map(lambda y: simulation(env, y), x))),

    toolbox.register("evaluate", evaluate)

#------------- b) Registering the EA operators --------------
    # 1. Standard operators
    toolbox.register("select", tools.selTournament,tournsize=tournament_size)
    # # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=bound_low, up=bound_up,eta=crowding_factor)
    # # toolbox.register("mutate", tools.mutPolynomialBounded, low=bound_low, up=bound_up,eta=crowding_factor,indpb=1.0/mutation_scaler)
    # toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # 2. Multiple operators
    # 2.1 All operators
    # 2.1.1 Crossover
    # Two-point: toolbox.register("mate", tools.cxTwoPoint)
    # Partially matched: toolbox.register("mate", tools.cxPartialyMatched)
    # Uniform: toolbox.register("mate", tools.cxUniform, indpb=0.05)
    # 2.1.2 Mutation
    # flip: toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # shuffle: toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    # uniformint: toolbox.register("mutate", tools.mutUniformInt(low = -1, high = 1, indpb = 0.05)

    if algorithm == 1:
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    elif algorithm == 2:
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    elif  algorithm == 3:
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low = -1, up = 1, indpb = 0.05)
    elif algorithm == 4:
        toolbox.register("mate", tools.cxBlend, alpha=0.05)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    elif algorithm == 5:
        toolbox.register("mate", tools.cxBlend, alpha=0.05)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    elif algorithm == 6:
        toolbox.register("mate", tools.cxBlend, alpha=0.05)
        toolbox.register("mutate", tools.mutUniformInt, low = -1, up = 1, indpb = 0.05)
    elif algorithm == 7:
        toolbox.register("mate", tools.cxUniform, indpb=0.05)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    elif algorithm == 8:
        toolbox.register("mate", tools.cxUniform, indpb=0.05)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    elif algorithm == 9:
        toolbox.register("mate", tools.cxUniform, indpb=0.05)
        toolbox.register("mutate", tools.mutUniformInt, low = -1, up = 1, indpb = 0.05)
    elif  algorithm == 10:
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
    elif  algorithm == 11:
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.15)
    elif algorithm == 12:
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.3)
    else:
        print("why?")

#---------------- Setting the game enviroment ----------------
    def simulation(env, x):
        f, p, e, t=env.play(pcont=x)
        return f, p
    # Plotting
    maxFitnessValues = []
    meanFitnessValues = []

    # Running the simulation
    def main(game, enemy):
        file_aux=open(experiment_name+'/results_enemy' + \
                          str(enemy) + str(algorithm) + '.txt', 'a')
        file_aux.write(f'\ngame {game} \n')
        file_aux.write('gen, best, mean, std, median, q1, q3, life')
        file_aux.close()

        #Creating the population
        pop = toolbox.populationCreator(n=pop_size) # Population is created as a list object
        pop_array = np.array(pop)
        generationCounter=0
        print("Start of evolution")

        # Evaluating all the population
        # fitnessValues=list(map(toolbox.evaluate, pop_array)) -> Won't work. Used Kamiel's
        fitnessValue = evaluate(pop_array)
        fitnessValue = fitnessValue[0].tolist()
        fitnesses = []
        lifes = []
        for value in  fitnessValue:
            fitnesses.append(value[0])
            lifes.append(value[1])
        for count, individual in enumerate(fitnesses):
            # Rewrites the fitness value in a way the DEAP algorithm can understand
            fitnesses[count] = (-individual, )

        # Assigning the fitness value to each individual
        for individual, fitnessValue in zip(pop, fitnesses):
            individual.fitness.values=fitnessValue

        # Extract each fitness value
        fitnessValues=[individual.fitness.values[0]
                       for individual in pop]

        # Saves first generation
        fits = fitnessValues
        g = generationCounter
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - abs(mean)**2)**0.5
        q1 = np.percentile(fits, 25)
        median = np.percentile(fits, 50)
        q3 = np.percentile(fits, 75)
        max_life = max(lifes)
        file_aux = open(experiment_name+'/results_enemy' +
                        str(enemy)+'Tournement.txt', 'a')
        file_aux.write(
            f'\n{str(g)}, {str(round(max(fits)))}, {str(round(mean,6))}, {str(round(std,6))}, {str(round(median,6))}, {str(round(q1,6))}, {str(round(q3,6))}, {str(round(max_life,6))}')
        file_aux.close()


        # Beggin the genetic loop
        # First, we start with the stopping condition
        while max(fitnessValues) < 100 and generationCounter < max_generations:
            begin_time = datetime.datetime.now()
            print("Being evolution time:", begin_time,"!!!")
            # Update generation counter
            generationCounter=generationCounter + 1
            print("-- Generation %i --" % generationCounter)

            # Begin genetic operators
            # 1. Selection: since we already defined the tournament before
            # we only need to select the population and its lenght
            # Selected individuals now will be in a list
            print("selection...")
            offspring=toolbox.select(pop, len(pop))
            for i in offspring:
                print(i.fitness.values[0])

            # Cloning the selected indv so we can apply the next genetic operators without affecting the original pop
            offspring=list(map(toolbox.clone, offspring))
            print("done")
            # 2. Crossover. Note taht the mate function takes two individuals as arguments and
            # modifies them in place, meaning they don't need to be reassigned
            print("Crossover...")
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < p_crossover:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            print("done")
            # 3. Mutation
            print("Mutation...")
            for mutant in offspring:
                # if random.random() < p_mutation:
                if random.random() < (1 - (generationCounter/max_generations)):
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Individuals that werent mutated remain intact, their fitness values don't need to eb recalculated
            # The rest of the individuals will have this value EMPTY
            # We now find those individuals and calculate the new fitness
            print("...re-evaluating fitness...")
            freshIndividuals=[ind for ind in offspring if not ind.fitness.valid]
            # Eval not work!!! :(( used Kamiels
            # freshFitnessValues=list(map(toolbox.evaluate, freshIndividuals))
            # for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            #     individual.fitness.values=fitnessValue
            pop_array = np.array(freshIndividuals)
            values = evaluate(pop_array)
            values = values[0].tolist()
            fitnesses = []
            for value in values:
                fitnesses.append(value[0])
                lifes.append(value[1])

            for count, individual in enumerate(fitnesses):
                fitnesses[count] = (individual, )

            for ind, fit in zip(freshIndividuals, fitnesses):
                ind.fitness.values = fit
            print("done")
            # Changes best individuals of population with worst individuals of the offspring
            amount_swithed_individuals=int(len(pop)/10)
            worst_offspring=deap.tools.selWorst(
                offspring, amount_swithed_individuals, fit_attr='fitness')
            best_gen=deap.tools.selBest(
                pop, amount_swithed_individuals, fit_attr='fitness')
            for count, individual in enumerate(worst_offspring):
                index=offspring.index(individual)
                offspring[index]=best_gen[count]

            # End of the proccess -> replace the old population wiht the new one
            pop[:]=offspring
            print(f"There are {len(pop)} individuals in the population ")

            # Gather all the fitnesses in one list and print the stats
            fits=[ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - abs(mean)**2)**0.5
            q1 = np.percentile(fits, 25)
            median = np.percentile(fits, 50)
            q3 = np.percentile(fits, 75)
            max_life = max(lifes)

            # For plotting
            maxFitness = max(fits)
            meanFitness = sum(fits)/len(pop)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            # Plot
            plt.plot(maxFitnessValues)
            plt.plot(meanFitnessValues)
            plt.ylabel("Values")
            plt.xlabel("Generations")
            plt.title(experiment_name + " game: " + str(game) + " " + "enemy: " + str(enemy))
            plt.savefig("adrian-testing" + "-algorithm-" + str(algorithm) +"-enemy-"+ str(enemy)+".png")
            plt.show()
            # DataFrame
            df.loc[int(algorithm)]=[min(fits),max(fits),mean,std]
            df.to_csv("adrian-testing" + "-algorithm-" + str(algorithm) +"-enemy-"+ str(enemy)+".csv", index=False)
            print("Saving...")
            # saves results for first pop
            file_aux=open(experiment_name+'/results_enemy' + \
                        str(enemy)+'Tournement.txt', 'a')
            file_aux.write(
                f'\n{str(g)}, {str(round(max(fits),6))}, {str(round(mean,6))}, {str(round(std,6))}, {str(round(median,6))}, {str(round(q1,6))}, {str(round(q3,6))}, {str(round(max_life,6))}')
            file_aux.close()
            print("Evolution ended in:", datetime.datetime.now() - begin_time)
            print("-- End of (successful) evolution --")
            best_ind=tools.selBest(pop, 1)[0]
            print("Best individual is %s, %s" %
                            (best_ind, best_ind.fitness.values))
            np.savetxt(experiment_name+'/best_game_'+str(game) + \
                            ',enemy_'+str(enemy)+'Tournement.txt', best_ind)
            print("Done. New generation...")
   
    main(game, enemy)
    plt.show()
    print("Run ended in:", datetime.datetime.now() - begin_game)

# 12 algorithms will be tested
# Tested 3 different crossover methods: Two-point, 6-point and Uniform.
# Tested 3 different mutation methods: Flip bit, Swap and Scramble
# Two sets of parameters, standard (what should produce good results according to previous results)
# and whack. The new algorithms wlll be tested with standard parameters and the old algorithm
# will be tested with 3 whack variations
### Unfortunatly DEAP doesn't have most of those methods so i used different ones
# 1. Two-point, Flip bit
# 2. Two-point, Swap
# 3. Two-point, Scramble
# 4. 6-point
# 5. 6-point
# 6. 6-point
# 7. Uni
# 8. Uni
# 9. Uni
# 10. Standard (two point = flipbit)

algorithms = [1,2,3,4,5,6,7,8,9,10,11,12]
enemies = [1,2,3,5,6,7,8]
for algorithm in algorithms:
    for enemy in enemies:
        for game in [1]:
            main1(game, enemy, algorithm)

print("End of script")
