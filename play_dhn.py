import os
import sys
import time
import chess
import random
import ray 
import numpy as np
import itertools as it

from uci import *
from evaluation import Evaluation
from coevolve import SynapticPopulation

import pickle
import neat
import neat.nn
#from pureples.shared.substrate import Substrate
#from pureples.shared.visualize import draw_net
#from pureples.hyperneat.hyperneat import create_phenotype_network
#from pureples.es_hyperneat.es_hyperneat import ESNetwork


#-----------------------------------------------------------
GENERATIONS = 1
POPSIZE = 16
MAX_DEPTH = 4
TIME_LIM = 3000
NUMBER_OF_MATCHES = 3
DIRECTORY = "."
#-----------------------------------------------------------

# # hyperneat params -----------------------------------------
# # S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
# VERSION = "M"
# VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

# # Create input layer coordinate map from specified input dimensions
# input_dimensions = [1, 64]
# x = np.linspace(-1.0, 1.0, input_dimensions[1]) if (input_dimensions[1] > 1) else [0.0]
# y = np.linspace(-1.0, 1.0, input_dimensions[0]) if (input_dimensions[0] > 1) else [-1.0]
# INPUT_COORDINATES = list(it.product(x,y))

# hidden_dimension = [1, 128]
# x = np.linspace(-1.0, 1.0, hidden_dimension[1]) if (hidden_dimension[1] > 1) else [0.0]
# y = np.linspace(-1.0, 1.0, hidden_dimension[0]) if (hidden_dimension[0] > 1) else [0.0]
# HIDDEN_COORDINATES = [list(it.product(x,y))]

# OUTPUT_COORDINATES = [(0.0, 1.0)]
# ACTIVATIONS = len(HIDDEN_COORDINATES) + 2

# SUBSTRATE = Substrate(
#     INPUT_COORDINATES, OUTPUT_COORDINATES, HIDDEN_COORDINATES)

# # for EShyperNEAT (avoid specifying the hidden node)
# #SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)
# #-----------------------------------------------------------


# # ES hyperneat params --------------------------------------

# NUMBER_OF_MATCHES = 1

# def params(version):
# 	"""
# 	ES-HyperNEAT specific parameters.
# 	"""
# 	return {"initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
# 		"max_depth": 1 if version == "S" else 2 if version == "M" else 3,
# 		"variance_threshold": 0.03,
# 		"band_threshold": 0.3,
# 		"iteration_level": 1,
# 		"division_threshold": 0.5,
# 		"max_weight": 5.0,
# 		"activation": "sigmoid"}

# DYNAMIC_PARAMS = params(VERSION)

# Config for CPPN.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
							neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
							'./config_neat')

def get_last_progress(directory, spec):
	"""
	Return the last progress done according to some spec
	"""
	generations = []
	files = os.listdir(directory)
	for file in files:
		if file.startswith(spec):
			splitted = file.split("-")[-1]
			generations.append(int(splitted))
	return max(generations) if generations != [] else 0

def get_last_checkpoints(directory, spec):
	""" 
	Return the filename of the last checkpoints file (synapses and bias matrices)
	"""
	lp = get_last_progress(directory, spec)
	if lp != 0:
		#lp = f"{'0'+str(lp) if len(str(lp))==1 else str(lp)}"
		last_checkpoint = f"{directory}/{spec}-{lp}"
		print(f"*** Last checkpoint found: {last_checkpoint}")
		return last_checkpoint 	
	else:
		print("*** No checkpoint found.")
		return None


@ray.remote
def play_single(max_depth, time_lim, player1, player2, netlist):
	"""
	(ray.remote function): to be used in parallel
	Play a single match given a SynapticPopulation and the two players indexes
	Args:
		max_depth:
		time_lim:
		player1: (int) index of the player1 net
		player2: (int) index of the player2 net
		netlist: (list) of net objects
	Return:
		result and board object
	"""
	# initialize
	s = time.time()
	result = [0]*2
	moves_counter = 1

	# extract net given the p1, p2 indexes
	net1, net2 = netlist[player1], netlist[player2]

	# initialize ES-hyper-NEAT players
	uci = UCI_EShypNEAT(
			max_depth = max_depth, 
			time_lim = time_lim, 
			net1 = net1,
			net2 = net2
			)

	# exit conditions / sum of material values
	exit_condition, material_value_diff_condition = False, False

	# play
	while not exit_condition and not material_value_diff_condition:
		t0 = time.time()
		try:
			print(f"{'-'*80}")
			print(f"{moves_counter}) - {'White turn:' if uci.board.turn else 'Black turn:'}")
			uci.board.push(uci.processCommand("go"))

			# estimate the exit condition
			material_value_diff_condition = abs(Evaluation.evaluate(uci.board)) >= 1500
			exit_condition = uci.board.is_checkmate() or uci.board.is_stalemate() or uci.board.can_claim_draw()
		except:
			break
		moves_counter += 1
		t1 = time.time()
		print(f"Elapsed (sec): {round(t1-t0,2)}", end="\n")

	# collect exit condition and save score
	if uci.board.is_checkmate():
		result = [1,0] if uci.board.outcome().winner else [0,1]
		print(f"Checkmate! {result}")

	elif uci.board.can_claim_draw() or uci.board.is_stalemate():
		result = [0.5,0.5]
		print("Draw!")

	elif material_value_diff_condition:
		diff = Evaluation.evaluate(uci.board)
		result = [1,0] if diff > 0 else [0,1]
		print(f"Win due to material value difference: {result} {diff}")

	e = time.time()
	print(f"Total execution time (sec): {round(e-s,2)}")

	return {"player1":player1,
			"player2":player2,
			"result":result, 
			"time":round(e-s,2),
			"board":uci.board}


def results2fitness(results):
	"""
	Convert the list of results from "play_single" into a vector of scores
	Args:
		results: (list) of each resulting outcome from "play_single"
	"""
	fitness = [0]*POPSIZE
	for match in results:
		fitness[match["player1"]] += match["result"][0]
		fitness[match["player2"]] += match["result"][1]
	return np.array(fitness).astype(float) 


def play_tournament(number_of_matches, netlist):
	"""
	Args:
		number_of_matches: (int) how many matches shall each subgenome play as White 
			(and as Black accordingly)
		netlist: (list) of all net objects
	"""
	tstart = time.time()

	# list all players
	allplayers = (list(range(0,len(netlist))))
	np.random.shuffle(allplayers)
	allplayers.extend(allplayers[0:number_of_matches])
	
	# save all combination of matches
	combinations = []
	for i in range(0, len(allplayers)-number_of_matches):
		tmp = []
		for j in range(number_of_matches):
			tmp.append((allplayers[i], allplayers[i+j+1]))
		combinations.append(tmp)

	# allocate a thread for each match 
	ids = []
	for i in range(len(combinations)):
		for j in range(len(combinations[i])):
			player1 , player2 = combinations[i][j][0], combinations[i][j][1]
			print(f"Now running in background: {player1, player2}")
			ids.append( play_single.remote(MAX_DEPTH, TIME_LIM, player1, player2, netlist) )

	results = ray.get(ids)

	tend = time.time()

	print("Tournament completed in (sec): ", round(tend-tstart,3))

	return results2fitness(results)


def eval_fitness(genomes, config):
	"""
	Fitness function.
	For each genome evaluate its fitness, in this case, as the mean squared error.
	"""
	indexlist = [i for i, _ in genomes][-POPSIZE:]
	netlist   = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]

	assert len(netlist) == POPSIZE, "Too much offsprings generated from past iteration!"
	
	new_fitness = play_tournament(NUMBER_OF_MATCHES, netlist)

	print(" *** Total amount of genomes:", max([i for i, _ in genomes]))
	print(" *** Current idx vector    : ", indexlist)
	print(" *** Current lenght of netlist:", len(netlist), end="\n\n")
	print(" *** Current fitness vector: ", new_fitness)

	for i, genome in genomes:
		if i in indexlist:
			genome.fitness = new_fitness[indexlist.index(i)] # i = 1, .., POPSIZE


def run(pop, gens):
	"""
	Create the population and run the task with eval_fitness as the fitness function.
	Returns the winning genome and the statistics of the run.
	Args:
		pop: (neat.Population) 
		gens: (int) number of generations
		version: (str) {"S", "M", "L"} (es hyperneat)
	"""
	stats = neat.statistics.StatisticsReporter()
	pop.add_reporter(neat.statistics.StatisticsReporter())
	pop.add_reporter(neat.reporting.StdOutReporter(True))
	pop.add_reporter(neat.Checkpointer(1))

	# ES hyper NEAt
	# global DYNAMIC_PARAMS
	# DYNAMIC_PARAMS = params(version)

	winner = pop.run(eval_fitness, gens)
	return winner, stats	


if __name__ == "__main__":

	t0total = time.time()

	# collect past generation if any (previous progress):
	last_checkpoint = get_last_checkpoints(DIRECTORY, "neat-checkpoint")

	# if you want to continue from past generations
	pop = neat.Population(CONFIG)

	if bool(last_checkpoint):
		pop = neat.Checkpointer.restore_checkpoint(last_checkpoint)

	# mutliprocessing: initialize
	ray.init(include_dashboard=False)

	# evolve
	winner, stats = run(pop, GENERATIONS)

	# mutliprocessing: close 
	ray.shutdown()

	t1total = time.time()
	print(f"Total job time elapsed (sec): {round(t1total-t0total,3)}")
