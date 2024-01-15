import os
import sys
import csv
import time
from stockfish import Stockfish

from uci import *
from evaluation import Evaluation
from coevolve import SynapticPopulation
from best_player_1hidd import initialize_syn_pop

# stockfish
STOCKFISH_ELO = 400
STOCKFISH_TIME_LIM = 50
TRIALS = 6 # number of matches to test again stockfish
STOCKFISH_PATH = "" # path to the stockfish executable

# UCI TEST engine params
TEST_DEPTH = 4
TEST_TIME_LIM = 3000

# load other parameters for the evochess engine
POPSIZE = 16
SINGLE_LAYER_N_NODES = [128]
NUMBER_OF_MATCHES = 3
MAX_DEPTH = 4
TIME_LIM = 3000
WEIGHT_METHOD = "unixavier"

SPEC      = f"pop{POPSIZE}_n{SINGLE_LAYER_N_NODES[0]}_m{NUMBER_OF_MATCHES}_d{MAX_DEPTH}_t{TIME_LIM}"
DIRECTORY = f"./population/{WEIGHT_METHOD}_{SPEC}"



def initialize_stockfish(stockfish_elo):
	"""
	initialize the stockfish engine according
	"""
	stockfish = Stockfish(path=STOCKFISH_PATH,
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
				#print(f"{'-'*80}\n{moves_counter}) - {'White turn:' if uci.board.turn else 'Black turn:'}")
				white_move = uci.processCommand("go")
				uci.board.push(white_move)
				stockfish.make_moves_from_current_position([white_move.__str__()])
				moves_counter_w += 1

				#display(uci.board)

				#print(f"{'-'*80}\n{moves_counter}) - {'White turn:' if uci.board.turn else 'Black turn:'}")
				black_move = stockfish.get_best_move_time(STOCKFISH_TIME_LIM)
				uci.board.push(chess.Move.from_uci(black_move))
				stockfish.make_moves_from_current_position([black_move])

				#print(stockfish.get_board_visual()) if printBoard else None
				#clear_output(wait=True); print("Evo as white:"); display(uci.board)

			# stockfish plays as White, evo as Black
			else:
				#print(f"{'-'*80}\n{moves_counter}) - {'White turn:' if uci.board.turn else 'Black turn:'}")
				white_move = stockfish.get_best_move_time(STOCKFISH_TIME_LIM)
				uci.board.push(chess.Move.from_uci(white_move))
				stockfish.make_moves_from_current_position([white_move])

				#display(uci.board)

				#print(f"{'-'*80}\n{moves_counter}) - {'White turn:' if uci.board.turn else 'Black turn:'}")
				black_move = uci.processCommand("go")
				uci.board.push(black_move)
				stockfish.make_moves_from_current_position([black_move.__str__()])
				moves_counter_b += 1

				#print(stockfish.get_board_visual()) if printBoard else None
				#clear_output(wait=True); print("Evo as white:"); display(uci.board) 

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
	print(f"Total execution time (sec): {round(e-s,2)}")

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


def test_multiple(sp, best_player_idx, n_match):
	""" Perform multiple tests according to n_match """

	tstart = time.time()

	# list all match
	combinations =  [True]*(n_match//2) + [False]*(n_match//2)
	print(combinations)

	ids = []
	for i in range(len(combinations)):

		stockfish = initialize_stockfish(STOCKFISH_ELO)

		uci = UCI(TEST_DEPTH,
				TEST_TIME_LIM,
				sp.matrix[:,best_player_idx], 
				sp.bias[:, best_player_idx],
				None,
				None,
				SINGLE_LAYER_N_NODES)

		print(f"Now running in background: is evo white? {combinations[i]}")
		#ids.append( test_single.remote(uci, stockfish, combinations[i], printBoard=False) )
		ids.append(test_single(uci, stockfish, combinations[i], printBoard=False))
	
	# results = ray.get(ids)

	tend = time.time()

	print("Tournament completed in (sec): ", round(tend-tstart,3))

	return results2score(ids), movesCounter(ids)


def save_into_file(score, move_counts, best_player_idx):

	with open("./evaluation/stockfish_eval.csv", "a", newline="") as csvfile:
		writer = csv.writer(csvfile, delimiter=",")
		writer.writerow([
			STOCKFISH_ELO,
			STOCKFISH_TIME_LIM,
			TRIALS,
			POPSIZE,
			SINGLE_LAYER_N_NODES[0],
			NUMBER_OF_MATCHES,
			TEST_DEPTH,
			TEST_TIME_LIM,
			WEIGHT_METHOD,
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

	if not os.path.exists("./evaluation/stockfish_eval.csv"):
		with open("./evaluation/stockfish_eval.csv", "w", newline="") as csvfile:
			writer = csv.writer(csvfile, delimiter=",")
			writer.writerow([
			"stockfish_elo",
			"stockfish_time_lim",
			"trials",
			"popsize",
			"nodes",
			"n_matches",
			"max_depth",
			"time_lim",
			"weights_init",
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

	PAST_SYN  = f"{DIRECTORY}/pop{POPSIZE}_n{SINGLE_LAYER_N_NODES[0]}_m{NUMBER_OF_MATCHES}_d{MAX_DEPTH}_t{TIME_LIM}_g{PAST_GEN}_synapses.npy"
	PAST_BIAS = f"{DIRECTORY}/pop{POPSIZE}_n{SINGLE_LAYER_N_NODES[0]}_m{NUMBER_OF_MATCHES}_d{MAX_DEPTH}_t{TIME_LIM}_g{PAST_GEN}_bias.npy"

	sp = initialize_syn_pop(POPSIZE, SINGLE_LAYER_N_NODES, PAST_SYN, PAST_BIAS, WEIGHT_METHOD)

	for individual_idx in range(sp.matrix.shape[1]):

		print(f"Now evaluating individual n: {individual_idx}")

		score, counter = test_multiple(sp, individual_idx, n_match = TRIALS)

		save_into_file(score, counter, individual_idx)

