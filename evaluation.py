# Credits:
# Disservin: https://github.com/Disservin/python-chess-engine/tree/master

from helpers import *
from psqt import *

import chess
import numpy as np

class Evaluation:

    @staticmethod
    def eval_side(board: chess.Board, color: chess.Color) -> int:
        occupied = board.occupied_co[color]

        material = 0
        psqt = 0

        # loop over all set bits
        while occupied:
            # find the least significant bit
            square = lsb(occupied)

            piece = board.piece_type_at(square)

            # add material
            material += piece_values[piece]

            # add piece square table value
            #psqt += (
            #    list(reversed(psqt_values[piece]))[square]
            #    if color == chess.BLACK
            #    else psqt_values[piece][square] )

            # remove lsb
            occupied = poplsb(occupied)

        return material + psqt

    @staticmethod
    def evaluate(board: chess.Board) -> int:
        return Evaluation.eval_side(board, chess.WHITE) - Evaluation.eval_side(
            board, chess.BLACK )


#--------------------------------------------------------------------
# custom: CoSYNE neuroevolution evaluation function: 
#-------------------------------------------------------------------- 

class CoevEvoEvaluation:

    @staticmethod
    def _fen2vect(board: chess.Board):
        """
        Return a vector representation of the current FEN board representation
        """
        rep = []
        fen = board.fen().split(" ")[0]
        fen = fen.replace("/","")
        for _,item in enumerate(fen):
            try:
                rep.extend([0]*int(item))
            except:
                rep.append(piece_fen_values[item])
        return np.array(rep)

    @staticmethod
    def _subgeno2mat(inputvect: np.array, subgeno: np.array, bias_geno: np.array, hnode: list):
        """
        Function which converts the single sub-genoma array into matrix(es) to be used in the shallow
        neural network implemented as below
        Args:
            inputvect: (np.array) input vector representation of pieces static value in the 
                current board
            subgeno: (np.array) encoding a flatten array of synapses to be converted in matrix(es)
            bias_geno: (np.array) bias term encoding
            hnode: (list) of integers representing how many hidden nodes are for each layer
            [128, 32] (two layers, 128 and 32 hiddend nodes respectively)
        Return a list of matrix(es) of the corresponding weights 
        """
        weights, biases = [], []

        # ensuring that only the copy is modified
        hnode_c = hnode.copy()

        # adding the input vector
        hnode_c.insert(0,inputvect.shape[0]) 

        # adding the final output node
        hnode_c.insert(len(hnode_c), 1) 

        past_index_n_hidden_s, past_index_n_hidden_b = 0,0

        # proceed with the vector to matrix(es) conversion
        for i,n_hidden in enumerate(hnode_c):
            if not i == 0:
                weights.append(
                    subgeno[ past_index_n_hidden_s : past_index_n_hidden_s+(hnode_c[i-1]*n_hidden) ]\
                    .reshape((n_hidden, hnode_c[i-1]), order="F")
                    )
                
                biases.append( 
                    bias_geno[ past_index_n_hidden_b : (past_index_n_hidden_b+hnode_c[i]) ]
                    )

                past_index_n_hidden_s = (past_index_n_hidden_s+(hnode_c[i-1]*n_hidden))
                past_index_n_hidden_b = (past_index_n_hidden_b+hnode_c[i])          

        return weights, biases 

    @staticmethod
    def activation_function(Z:np.array, act_fun: str):
        """
        Return the activation function of Z = W_[l] × A_[l-1]
        """
        if act_fun == "relu":
            return np.where(Z > 0, Z, 0)

        elif act_fun == "sigmoid":
            return 1/(1+np.exp(Z)) 

        elif act_fun == "tanh":
            return (np.exp(Z)-np.exp(-Z)) / (np.exp(Z)+np.exp(-Z))

    @staticmethod
    def eval_NN(board: chess.Board, subgeno: np.array, bias_geno: np.array, hnode: list) -> float:
        """
        Implementation of the CoSYNE neuroevolution evaluation function:
        Feedforward pass only from synapses build upon neuroevolution subgenes
        """
        A = CoevEvoEvaluation._fen2vect(board)
        weights, biases = CoevEvoEvaluation._subgeno2mat(A, subgeno, bias_geno, hnode)

        # iterate over the feedforward pass
        for i, W in enumerate(weights):
            if not biases[i].shape[0]==0:
                Z = np.dot(W, A) + biases[i]
            else:
                Z = np.dot(W, A)
            A = CoevEvoEvaluation.activation_function(Z, "sigmoid" if i!=len(weights)-1 else "tanh")
        
        if A[0] == None or A[0] == np.nan:
            A[0] = 0
            print("Some exception is found!!")
        return A[0]
     
    @staticmethod
    def eval_side(board: chess.Board, color: chess.Color) -> int:
        occupied = board.occupied_co[color]
        material = 0
        # loop over all set bits
        while occupied:
            # find the least significant bit
            square = lsb(occupied)

            piece = board.piece_type_at(square)

            # add material
            material += piece_values[piece]

            # remove lsb
            occupied = poplsb(occupied)

        return material

    @staticmethod
    def evaluate(board: chess.Board, subgeno: np.array, bias_geno: np.array, hnode: list) -> float:

        material = (CoevEvoEvaluation.eval_side(board, chess.WHITE) - CoevEvoEvaluation.eval_side(
            board, chess.BLACK )) / 900

        functional = CoevEvoEvaluation.eval_NN(board, subgeno, bias_geno, hnode)

        return material + functional
        

#--------------------------------------------------------------------
# custom: ES-hyperNEAT neuroevolution evaluation function: 
#-------------------------------------------------------------------- 

class EShyperNEATEvaluation:

    @staticmethod
    def _fen2vect(board: chess.Board):
        """
        Return a vector representation of the current FEN board representation
        """
        rep = []
        fen = board.fen().split(" ")[0]
        fen = fen.replace("/","")
        for _,item in enumerate(fen):
            try:
                rep.extend([0]*int(item))
            except:
                rep.append(piece_fen_values[item])
        return np.array(rep)

    @staticmethod
    def eval_side(board: chess.Board, color: chess.Color) -> int:
        occupied = board.occupied_co[color]
        material = 0
        # loop over all set bits
        while occupied:
            # find the least significant bit
            square = lsb(occupied)

            piece = board.piece_type_at(square)

            # add material
            material += piece_values[piece]

            # remove lsb
            occupied = poplsb(occupied)

        return material

    @staticmethod
    def evaluate(board: chess.Board, net) -> float: 
        """
        Return the feedforward activation from the fen2vect board representation
        """
        material = (EShyperNEATEvaluation.eval_side(board, chess.WHITE) - EShyperNEATEvaluation.eval_side(
            board, chess.BLACK )) / 900

        functional = net.activate(EShyperNEATEvaluation._fen2vect(board))[0]

        return material + functional
