# Neuroevolved Chess Engine:

An attempt to evolve a chess engine tuning chess evaluation functions evolved by Neuroevolution frameworks (CoSyNE and NEAT). 

Three architectures had been evolved for 200 generations:
1. CoSyNE-based fixed topology: 1 hidden layer architecture `n = {64, 128, 1}`.
2. CoSyNE-based fixed topology: 2 hidden layer architectures `n = {64, 64, 64, 1}`.
3. NEAT-based varying network topology: 1 hidden layer architecture `n = {64, 128, 1}`.

The project is still *under development*: Neuroevolved virtual players have partially learned the optimization task and more work is needed in order to have them fully able to play properly.

--- 

## How to test:
1. Install all required libraries in a dedicated python 3.10 environment by: `pip install -r requirements.txt`.
2. Download and install a Stockfish executable chess engine in your local machine: (https://stockfishchess.org/).
3. Choose the `PAST_GEN` generation to be tested by modifying the global var.
4. Adapt the `STOCKFISH_PATH` global var pointing at to your local Stockfish file in `test_*.py` and files if you want to test the engine against Stockfish: results will be saved in `/evaluation/stockfish_eval.csv` if playing with `_1hidd` or `_2hidd` versions otherwise if you need to test NEAT based chess engine, run `test_dhn.py`: results will be stored in `/evaluation/stockfish_eval_neat.csv`. You can run the testing procedure by specifying the number of the generation to be tested:

```shell
python test_1hidd.py 150
```
Generations available: (25, 50, 75, 100, 125, 150, 175, 200).

---

## Visual assessment:
- Adapt the `STOCKFISH_PATH` global var pointing at to your local Stockfish file in `visual_test.ipynb` and execute the codecell.

Currently, you can test only the 1 hidden layer architecture by specifying the `PAST_GEN` generation.

---

## To do:
* [x] Evolution procedure (multi-thread spawning matches to perform tournament between individual of the current population)
* [x] Testing procedure (single-thread running due to Stockfish limitation)
* [ ] Evolving NEAT virtual players by enforcing elitims (currently disabled)
* [ ] Visual assessment of 2-hidden layers architecture and NEAT chess based engine
* [ ] Re-implementing the engine (further implementations improvements to be defined for better performance)

---

# Credits:
* **Disservin**: python chess engine (GNU GPL-3.0 Licence) (https://github.com/Disservin/python-chess-engine/tree/master)
* **Ilya Zhelyabuzhsky**: Stockfish python API (MIT Licence) (https://github.com/zhelyabuzhsky/stockfish/tree/master)
* **Official Stockfish** (GNU GPL-3.0 Licence): (https://github.com/official-stockfish/Stockfish)

# Tested with:
* Python 3.10
* Ubuntu 22.04 LTS
* Stockfish v.16
