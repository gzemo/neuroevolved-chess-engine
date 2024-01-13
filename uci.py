import search as Search
import evaluation as Eval
from helpers import *
from limits import *

# External
from sys import stdout
from threading import Thread
import chess
import numpy as np

class ThreadWithReturnValue(Thread):
    """
    Subclass of Thread object to handle the best move to be returned 
    in order to be further used
    """
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class UCI:
    def __init__(self,
        max_depth:int,
        time_lim:int,
        subgeno_p1: np.array,
        bias_geno_p1: np.array, 
        subgeno_p2: np.array,
        bias_geno_p2: np.array, 
        hnode: list) -> None:

        # Parameters
        self.max_depth = max_depth
        self.time_lim = time_lim
        self.subgeno_p1 = subgeno_p1
        self.bias_geno_p1 = bias_geno_p1
        self.subgeno_p2 = subgeno_p2
        self.bias_geno_p2 = bias_geno_p2
        self.hnode = hnode

        self.out = stdout
        self.board = chess.Board()
        #self.search = Search.Search(self.board, self.subgeno_p1, self.bias_geno_p1, self.hnode)
        self.search_p1 = Search.Search(self.board, self.subgeno_p1, self.bias_geno_p1, self.hnode, None)
        self.search_p2 = Search.Search(self.board, self.subgeno_p2, self.bias_geno_p2, self.hnode, None)
        self.thread: Thread | None = None

    def output(self, s) -> None:
        self.out.write(str(s) + "\n")
        self.out.flush()

    def stop(self) -> None:
        self.search.stop = True
        if self.thread is not None:
            try:
                self.thread.join()
            except:
                pass

    def quit(self) -> None:
        self.search.stop = True
        if self.thread is not None:
            try:
                self.thread.join()
            except:
                pass

    def uci(self) -> None:
        self.output("id name python-chess-engine")
        self.output("id author Max, aka Disservin")
        self.output("")
        self.output("option name Move Overhead type spin default 5 min 0 max 5000")
        self.output("option name Ponder type check default false")
        self.output("uciok")

    def isready(self) -> None:
        self.output("readyok")

    def ucinewgame(self) -> None:
        pass

    def eval(self) -> None:
        # eval = Eval.CoevEvoEvaluation()
        # self.output(eval.evaluate(self.board))
        raise Exception("currently disabled...")

    def update_board(self, board:chess.Board):
        """ 
        Custom made function to update the search given a certain board
        """
        self.board = board

    def processCommand(self, input: str) -> None:
        splitted = input.split(" ")

        # (python 3.10 needed here due to the match case instance)
        match splitted[0]:
            case "quit":
                self.quit()
            case "stop":
                self.stop()
                self.search.reset()
            case "ucinewgame":
                self.ucinewgame()
                self.search.reset()
            case "uci":
                self.uci()
            case "isready":
                self.isready()
            case "setoption":
                pass
            case "position":
                self.search.reset()
                fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                movelist = []

                move_idx = input.find("moves")
                if move_idx >= 0:
                    movelist = input[move_idx:].split()[1:]

                if splitted[1] == "fen":
                    position_idx = input.find("fen") + len("fen ")

                    if move_idx >= 0:
                        fen = input[position_idx:move_idx]
                    else:
                        fen = input[position_idx:]

                self.board.set_fen(fen)
                self.search.hashHistory.clear()

                for move in movelist:
                    self.board.push_uci(move)
                    self.search.hashHistory.append(self.search.getHash())

            case "print":
                print(self.board)

            case "eval":
                return self.eval()

            case "go":
                #limits = Limits(0, MAX_PLY, 0)
                limits = Limits(0, self.max_depth, self.time_lim) 

                l = ["depth", "nodes"]
                for limit in l:
                    if limit in splitted:
                        limits.limited[limit] = int(splitted[splitted.index(limit) + 1])

                ourTimeStr = "wtime" if self.board.turn == chess.WHITE else "btime"
                ourTimeIncStr = "winc" if self.board.turn == chess.WHITE else "binc"

                if ourTimeStr in input:
                    limits.limited["time"] = (
                        int(splitted[splitted.index(ourTimeStr) + 1]) / 20
                    )

                if ourTimeIncStr in input:
                    limits.limited["time"] += (
                        int(splitted[splitted.index(ourTimeIncStr) + 1]) / 2
                    )

                #self.search.limit = limits
                self.search_p1.limit = limits
                self.search_p2.limit = limits

                # default implementation with no return
                #self.thread = Thread(target=self.search.iterativeDeepening)
                #self.thread.start()

                # implementing a custom Thread function in order to retrieve the best move:
                #self.thread = ThreadWithReturnValue(target=self.search.iterativeDeepening)
                #self.thread.start()
                #bestmove = self.thread.join()
                
                # implementing a custom Thread function that switch between players
                # if player 2 synapses are None consider the first player
                if self.board.turn or self.subgeno_p2 == None: 
                    self.thread = ThreadWithReturnValue(
                        target = self.search_p1.iterativeDeepening)
                else:
                    self.thread = ThreadWithReturnValue(
                        target = self.search_p2.iterativeDeepening)
                
                self.thread.start()
                bestmove = self.thread.join()

                return bestmove



class UCI_EShypNEAT:
    def __init__(self,
        max_depth:int, 
        time_lim:int, 
        net1, net2) -> None:

        # Parameters
        self.max_depth = max_depth
        self.time_lim = time_lim
        self.net1 = net1
        self.net2 = net2

        self.out = stdout
        self.board = chess.Board()

        # ES-hyper NEAT neuroevolution framework
        self.search_p1 = Search.Search(self.board, None, None, None, self.net1)
        self.search_p2 = Search.Search(self.board, None, None, None, self.net2)

        self.thread: Thread | None = None

    def output(self, s) -> None:
        self.out.write(str(s) + "\n")
        self.out.flush()

    def stop(self) -> None:
        self.search.stop = True
        if self.thread is not None:
            try:
                self.thread.join()
            except:
                pass

    def quit(self) -> None:
        self.search.stop = True
        if self.thread is not None:
            try:
                self.thread.join()
            except:
                pass

    def uci(self) -> None:
        self.output("id name python-chess-engine")
        self.output("id author Max, aka Disservin")
        self.output("")
        self.output("option name Move Overhead type spin default 5 min 0 max 5000")
        self.output("option name Ponder type check default false")
        self.output("uciok")

    def isready(self) -> None:
        self.output("readyok")

    def ucinewgame(self) -> None:
        pass

    def eval(self) -> None:
        raise Exception("currently disabled...")

    def update_board(self, board:chess.Board):
        """ 
        Custom made function to update the search given a certain board
        """
        self.board = board

    def processCommand(self, input: str) -> None:
        splitted = input.split(" ")

        # (python 3.10 needed here due to the match case instance)
        match splitted[0]:
            case "quit":
                self.quit()
            case "stop":
                self.stop()
                self.search.reset()
            case "ucinewgame":
                self.ucinewgame()
                self.search.reset()
            case "uci":
                self.uci()
            case "isready":
                self.isready()
            case "setoption":
                pass
            case "position":
                self.search.reset()
                fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                movelist = []

                move_idx = input.find("moves")
                if move_idx >= 0:
                    movelist = input[move_idx:].split()[1:]

                if splitted[1] == "fen":
                    position_idx = input.find("fen") + len("fen ")

                    if move_idx >= 0:
                        fen = input[position_idx:move_idx]
                    else:
                        fen = input[position_idx:]

                self.board.set_fen(fen)
                self.search.hashHistory.clear()

                for move in movelist:
                    self.board.push_uci(move)
                    self.search.hashHistory.append(self.search.getHash())

            case "print":
                print(self.board)

            case "eval":
                return self.eval()

            case "go":
                #limits = Limits(0, MAX_PLY, 0)
                limits = Limits(0, self.max_depth, self.time_lim) 

                l = ["depth", "nodes"]
                for limit in l:
                    if limit in splitted:
                        limits.limited[limit] = int(splitted[splitted.index(limit) + 1])

                ourTimeStr = "wtime" if self.board.turn == chess.WHITE else "btime"
                ourTimeIncStr = "winc" if self.board.turn == chess.WHITE else "binc"

                if ourTimeStr in input:
                    limits.limited["time"] = (
                        int(splitted[splitted.index(ourTimeStr) + 1]) / 20
                    )

                if ourTimeIncStr in input:
                    limits.limited["time"] += (
                        int(splitted[splitted.index(ourTimeIncStr) + 1]) / 2
                    )

                #self.search.limit = limits
                self.search_p1.limit = limits
                self.search_p2.limit = limits

                if self.board.turn or self.net2 == None: 
                    self.thread = ThreadWithReturnValue(
                        target = self.search_p1.iterativeDeepening)
                else:
                    self.thread = ThreadWithReturnValue(
                        target = self.search_p2.iterativeDeepening)
                
                self.thread.start()
                bestmove = self.thread.join()

                return bestmove

# np.random.seed(111)

# hnode = [128]

# subgeno_p1 = np.random.uniform(-1,1, size = (hnode[0]*64+hnode[0]))
# bias_geno_p1 = np.random.uniform(-1, 1, size = hnode[0])
# subgeno_p2 = np.random.uniform(-1,1, size = (hnode[0]*64+hnode[0]))
# bias_geno_p2 = np.random.uniform(-1, 1, size = hnode[0])

# uci = UCI(4,3000,subgeno_p1, bias_geno_p1, subgeno_p2, bias_geno_p2, hnode)

# #uci.processCommand("go")
# c = 0
# while True:
#     print(f"{'-'*80},\n turn: {c}")
#     uci.board.push(uci.processCommand("go"))
#     print(uci.board)
#     c += 1
