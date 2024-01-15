# Neuroevolved chess engine:

An attempt to evolve a chess engine tuning chess evaluation functions evolved by Neuroevolution frameworks (CoSyNE and NEAT). 

# How to test:
1. Install all required libraries in a dedicated python 3.10 environment by: `pip install -r requirements.txt`
2. Gather and install a Stockfish executable chess engine in your local machine: (https://stockfishchess.org/)
3. Adapt the `STOCKFISH_PATH` global var pointing at to your local Stockfish file.
4. Run `visual_test.ipynb`
 
# Credits:
* Ilya Zhelyabuzhsky: Stockfish python API (https://github.com/zhelyabuzhsky/stockfish/tree/master)
* Official Stockfish (GNU GPL-3.0 Licence): (https://github.com/official-stockfish/Stockfish)
* Disservin: python chess engine (https://github.com/Disservin/python-chess-engine/tree/master)

# Tested with:
* Python 3.10
* Ubuntu 22.04 LTS
* Stockfish v.16
