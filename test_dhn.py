import os
import sys
import csv
import time
from stockfish import Stockfish

from uci import *
from evaluation import Evaluation

import pickle
import neat
import neat.nn

# stockfish
STOCKFISH_ELO = 400
STOCKFISH_TIME_LIM = 50
TRIALS = 6 # number of matches to test again stockfish

# UCI TEST engine params
TEST_DEPTH = 4
TEST_TIME_LIM = 3000

# # load other parameters for the evochess engine
POPSIZE = 16
# SINGLE_LAYER_N_NODES = [128]
NUMBER_OF_MATCHES = 3
MAX_DEPTH = 4
TIME_LIM = 3000
# WEIGHT_METHOD = "unixavier"

# SPEC      = f"pop{POPSIZE}_n{SINGLE_LAYER_N_NODES[0]}_m{NUMBER_OF_MATCHES}_d{MAX_DEPTH}_t{TIME_LIM}"
DIRECTORY = f"./population/neat-checkpoints"

# Config for CPPN.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
							neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
							'./config_neat')



def initialize_stockfish(stockfish_elo):
	"""
	initialize the stockfish engine according
	"""
	stockfish_path = "./stockfish/stockfish-ubuntu-x86-64-avx2"
	stockfish = Stockfish(path=stockfish_path,
		depth = 4,
		parameters = {"Threads": 1, "Hash": 0})
	stockfish.set_elo_rating(stockfish_elo)
	return stockfish


def retrieve_best_player(generation):
	"""
	given a past gen retrieve from the csv of best_player the idx of the best synapses
	"""
	with open("./evaluation/best_player.csv", "r") as csvfile:
		reader = csv.reader(csvfile, delimiter=",")
		header = reader.__next__()

		for i,row in enumerate(reader):
			print(row)
			if row[header.index("generation")] == generation:
				return int(row[header.index("best_player_idx")])

		raise Exception("Best player of that generation not tested yet!")


#@ray.remote
def test_single(uci, stockfish, isWhiteEvo, printBoard=False):
	"""
	Perform a single match between evochess and stockfish:
	Args:
		uci: uci object loaded with evolutionary derived synapses (evochess engine)
		stockfish: valid stockfish engine (stockfish.models.Stockfish)
		isWhiteEvo: bool, True if evochess plays as White, False otherwise
	"""
	s = time.time()
	result = [0]*2
	moves_counter_w , moves_counter_b = 0,0

	# exit conditions / sum of material values
	exit_condition, material_value_diff_condition = False, False

	# play
	while not exit_condition and not material_value_diff_condition:
		
		t0 = time.time()
		
		try:
			# evo plays as White, stockfish as Black
			if isWhiteEvo:
				white_move = uci.processCommand("go")
				uci.board.push(white_move)
				stockfish.make_moves_from_current_position([white_move.__str__()])
				moves_counter_w += 1

				#clear_output(wait=True); print("Evo as white:"); display(uci.board)

				black_move = stockfish.get_best_move_time(STOCKFISH_TIME_LIM)
				uci.board.push(chess.Move.from_uci(black_move))
				stockfish.make_moves_from_current_position([black_move])

				#clear_output(wait=True); print("Evo as white:"); display(uci.board)

			# stockfish plays as White, evo as Black
			else:
				white_move = stockfish.get_best_move_time(STOCKFISH_TIME_LIM)
				uci.board.push(chess.Move.from_uci(white_move))
				stockfish.make_moves_from_current_position([white_move])

				#clear_output(wait=True); print("Evo as black:"); display(uci.board) 

				black_move = uci.processCommand("go")
				uci.board.push(black_move)
				stockfish.make_moves_from_current_position([black_move.__str__()])
				moves_counter_b += 1

				#clear_output(wait=True); print("Evo as black:"); display(uci.board) 

			# estimate the exit condition
			material_value_diff_condition = abs(Evaluation.evaluate(uci.board)) >= 1500
			exit_condition = uci.board.is_checkmate() or uci.board.is_stalemate() or uci.board.can_claim_draw()
		
		except:
			break

		t1 = time.time()

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
	#print(f"Total execution time (sec): {round(e-s,2)}")
	#print(uci.board)

	return {"evo_player_result":result[0] if isWhiteEvo else result[1],
			"stockfish_result" :result[1] if isWhiteEvo else result[0],
			"time":round(e-s,2),
			"board":uci.board,
			"moves_counter_w": moves_counter_w,
			"moves_counter_b": moves_counter_b}


def movesCounter(results):
	""" Converts the number of moves made """
	w_count, b_count = 0,0
	for re in results:
		w_count += re["moves_counter_w"]
		b_count += re["moves_counter_b"]
	return {"w_count": w_count, "b_count": b_count, "overall": w_count+b_count}


def results2score(results):
	""" Convert the result into score """
	win, draw, loss = 0,0,0
	for re in results:
		if   re["evo_player_result"] == 1:
			win += 1
		elif re["evo_player_result"] == 0.5:
			draw += 1
		elif re["evo_player_result"] == 0:
			loss += 1 
	return {"win":win, "draw":draw, "loss":loss}


def test_multiple(net, n_match):
	""" Perform multiple tests according to n_match """

	tstart = time.time()

	# list all match
	combinations =  [True]*(n_match//2) + [False]*(n_match//2)
	print(combinations)

	ids = []
	for i in range(len(combinations)):

		stockfish = initialize_stockfish(STOCKFISH_ELO)

		uci = UCI_EShypNEAT(TEST_DEPTH,
							TEST_TIME_LIM,
							net,
							None)

		print(f"Now running in background: is evo white? {combinations[i]}")
		#ids.append( test_single.remote(uci, stockfish, combinations[i], printBoard=False) )
		ids.append(test_single(uci, stockfish, combinations[i], printBoard=False))
	
	# results = ray.get(ids)

	tend = time.time()

	print("Tournament completed in (sec): ", round(tend-tstart,3))

	return results2score(ids), movesCounter(ids)


def save_into_file(score, move_counts, best_player_idx):

	with open("./evaluation/stockfish_eval_neat.csv", "a", newline="") as csvfile:
		writer = csv.writer(csvfile, delimiter=",")
		writer.writerow([
			STOCKFISH_ELO,
			STOCKFISH_TIME_LIM,
			TRIALS,
			POPSIZE,
			NUMBER_OF_MATCHES,
			TEST_DEPTH,
			TEST_TIME_LIM,
			PAST_GEN,
			best_player_idx,
			score["win"],
			score["draw"],
			score["loss"],
			move_counts["w_count"],
			move_counts["b_count"],
			move_counts["overall"],
			])


if __name__=="__main__":

	if not os.path.exists("./evaluation/stockfish_eval_neat.csv"):
		with open("./evaluation/stockfish_eval_neat.csv", "w", newline="") as csvfile:
			writer = csv.writer(csvfile, delimiter=",")
			writer.writerow([
			"stockfish_elo",
			"stockfish_time_lim",
			"trials",
			"popsize",
			"n_matches",
			"max_depth",
			"time_lim",
			"generation",
			"best_player_idx",
			"win",
			"draw",
			"loss",
			"w_count",
			"b_count",
			"overall"
			])

	assert len(sys.argv) == 2, "Arguments to be pass must be equal to 1: (generation)"
	
	PAST_GEN  = str(sys.argv[1]) if len(str(sys.argv[1]))>1 else "0"+str(sys.argv[1])
	print(PAST_GEN)

	PAST_CKP  = f"{DIRECTORY}/neat-checkpoint-{PAST_GEN}"

	pop = neat.Population(CONFIG)
	pop = neat.Checkpointer.restore_checkpoint(PAST_CKP).population

	for individual_idx in list(pop.keys()):

		print(f"Now evaluating individual n: {individual_idx}")

		net = neat.nn.FeedForwardNetwork.create(pop[individual_idx], CONFIG)

		score, counter = test_multiple(net, n_match = TRIALS)

		save_into_file(score, counter, individual_idx)

