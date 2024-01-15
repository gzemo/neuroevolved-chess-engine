import os
import sys
import time
import chess
import ray 
import numpy as np

from uci import *
from evaluation import Evaluation
from coevolve import SynapticPopulation

#-----------------------------------------------------------
POPSIZE = 16
SINGLE_LAYER_N_NODES = [64,64]
NUMBER_OF_MATCHES = 3
MAX_DEPTH = 4
TIME_LIM = 3000
GENERATIONS = 31

SPEC      = f"pop{POPSIZE}_n{SINGLE_LAYER_N_NODES[0]}_{SINGLE_LAYER_N_NODES[1]}_m{NUMBER_OF_MATCHES}_d{MAX_DEPTH}_t{TIME_LIM}"
DIRECTORY = f"./population/unixavier_{SPEC}"

OUTDIR = f"{DIRECTORY}/{SPEC}"
#-----------------------------------------------------------


def get_last_progress(directory, spec):
	"""
	Return the last progress done according to some spec
	"""
	generations = []
	files = os.listdir(directory)
	for file in files:
		if file.startswith(spec):
			splitted = file.split("/")[-1].split("_")
			for i,feature in enumerate(splitted):
				if feature.startswith("g"):
					generations.append(int(feature[1:]))
					break
	return max(generations) if generations != [] else 0

def get_last_checkpoints(directory, spec):
	""" 
	Return the filename of the last checkpoints file (synapses and bias matrices)
	"""
	lp = get_last_progress(directory, spec)
	if lp != 0:
		lp = f"{'0'+str(lp) if len(str(lp))==1 else str(lp)}"
		print("***",lp)
		return (
			f"{directory}/{spec}_g{lp}_synapses.npy", 
			f"{directory}/{spec}_g{lp}_bias.npy"
			) 
	else:
		return None,None


@ray.remote
def play_single(sp, max_depth, time_lim, player1, player2):
	"""
	(ray.remote function): to be used in parallel
	Play a single match given a SynapticPopulation and the two players indexes
	Return:
		result and board object
	"""
	# initialize
	s = time.time()
	result = [0]*2
	moves_counter = 1
	uci = UCI(max_depth = max_depth,
			time_lim = time_lim,
			subgeno_p1  = sp.matrix[:,player1],
			bias_geno_p1 = sp.bias[:,player1], 
			subgeno_p2   = sp.matrix[:,player2],
			bias_geno_p2 = sp.bias[:,player2],
			hnode = SINGLE_LAYER_N_NODES ) 


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
	return np.array(fitness)


def play_tournament(sp, number_of_matches = 4):
	"""
	Args:
		sp: SynapticPopulation instance to be tested
		number_of_matches: (int) how many matches shall each subgenome play as White 
			(and as Black accordingly)
	"""
	tstart = time.time()

	# list all players
	allplayers = (list(range(0,POPSIZE))*2)[0:POPSIZE+number_of_matches]

	# draw matches
	matches = [(i, allplayers[(i+1) : (i+number_of_matches+1)]) for i in range(POPSIZE)]

	# save all combination of matches
	combinations = [[ (matches[i][0], matches[i][1][j]) for j in range(len(matches[i][1])) ] for i in range(len(matches))]

	# allocate a thread for each match 
	ids = []
	for i in range(len(combinations)):
		for j in range(len(combinations[i])):
			print(f"Now running in background: {combinations[i][j][0], combinations[i][j][1]}")
			ids.append( play_single.remote(sp, MAX_DEPTH, TIME_LIM, combinations[i][j][0], combinations[i][j][1]) )

	results = ray.get(ids)

	tend = time.time()

	print("Tournament completed in (sec): ", round(tend-tstart,3))

	return results2fitness(results)


def evo(sp, generations:int):
	"""
	Evolve over generations:
	Args:
		sp: (SynapticPopulation) 
		generations: (int) number of generation to be evolved
	"""
	initial_gen = sp.generation

	for _ in range(1, generations+1):

		print(f"{'='*80}")
		print(f" Now evolving generation: {initial_gen + _}" )
		print(f"{'='*80}", end="\n\n")

		new_fitness = play_tournament(sp, number_of_matches=NUMBER_OF_MATCHES)

		sp.evolve(new_fitness)

		randomidx = np.arange(0,POPSIZE)
		np.random.shuffle(randomidx)
		sp.reshufle_order(randomidx)

		sp.save_progress(OUTDIR)


if __name__ == "__main__":

	t0total = time.time()

	# initialize synaptic population
	sp = SynapticPopulation(popsize=POPSIZE,
		hnode = SINGLE_LAYER_N_NODES,
		n_synapses=64*SINGLE_LAYER_N_NODES[0]+SINGLE_LAYER_N_NODES[0]*SINGLE_LAYER_N_NODES[1]+ SINGLE_LAYER_N_NODES[1],
		n_bias=SINGLE_LAYER_N_NODES[0]+SINGLE_LAYER_N_NODES[1],
		weightmethod="unixavier")


	# collect past generation if any (previous progress):
	past_syn, past_bias = get_last_checkpoints(DIRECTORY, SPEC)

	# if you want to continue from past generations
	if bool(past_syn) and bool(past_bias):
		sp.load_matrix(past_syn)
		sp.load_bias(past_bias)

	# mutliprocessing: initialize
	ray.init(include_dashboard=False)

	# evolve
	evo(sp, generations=GENERATIONS)

	# mutliprocessing: close 
	ray.shutdown()

	t1total = time.time()
	print(f"Total job time elapsed (sec): {round(t1total-t0total,3)}")
