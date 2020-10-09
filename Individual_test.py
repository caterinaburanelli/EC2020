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

def main1(seed, game ,algorithm, group, algorithm_number, gen):
    experiment_name = 'dummy_demo'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize the amout of neurons
    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[1,2,3,4,5,6,7,8],
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


    creator.create("FitnessMin", base.Fitness, weights=(30.0, 30.0))
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
    toolbox.register("mate", tools.cxBlend)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament)

    creator.create("FitnessMin", base.Fitness, weights=(100,))

    #----------


    def simulation(env,x):
        f,p,e,t = env.play(pcont=x)
        return f, p
  
    a =  f'\Algorithm_{algorithm_number}_group_{group}\individuals_game_{game}\game_{game}_gen_{gen}_group_{group}_Tournement.txt'
    print(f"game = {game}, gen = {gen}")
    # Load specialist controller
    sol = np.loadtxt('dummy_demo' + a)
    print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY \n')
    print(evaluate([sol]))
    enemy_life = env.get_list_enemy_life()
    fitnesses = env.get_list_fitnesses()
    competition = env.get_lists_competition_life()

    sum_life = np.sum(competition[0])
    sum_time = np.sum(competition[1])

    if enemy_life.count(0) == 4:
        file_aux  = open('champs.txt','a')
        file_aux.write(f"Algorithm = {algorithm_number}, Group = {group}, Game = {game}, Gen = {gen} sum_player_life = {str(sum_life)}, mean_enemy_life = {np.mean(enemy_life)}, sum_time = {sum_time}, mean_fitnesses = {np.mean(fitnesses)}\n")
        file_aux.close()
    elif enemy_life.count(0) == 5:
        file_aux  = open('super_champs.txt','a')
        file_aux.write(f"Algorithm = {algorithm_number}, Group = {group}, Game = {game}, Enemy_lifes = {str(enemy_life)}, fitnesses = {fitnesses}\n")
        file_aux.close()




algorithm = 'Tournement'
algorithm_number = ["1", "2"]
groups = ["1", "2"]

for group in groups:
    for game in range(10):
        for gen in range(5,16):
            seed = random.randint(1, 126)
            main1(seed, game ,algorithm, group, algorithm_number, gen)