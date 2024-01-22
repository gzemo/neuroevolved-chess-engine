import time
import random
import numpy as np

class SynapticPopulation:
	"""
	Handling the representation and evolution of the synaptic population
	"""
	def __init__(self, popsize:int,
		n_synapses:int,
		n_bias:int,
		hnode:list = [128],
		weightmethod:str = "unixavier",
		weightlim:tuple = (-5, 5),
		bias_weightlim:tuple = (-5, 5),
		cross_prob:float = 0.6, 
		swap_prob:float = 0.3, 
		cross_strategy:str = "uniform", 
		mut_prob:float = 0.6, 
		mut_sigma:float = 0.01,
		mut_decay:bool = True,
		mut_decay_factor:float = 0.995,
		perm_prob:float = 0.5,
		perm_adaptive:bool = True):
		"""
		Args:
			popsize: int, number of instances to evolve
			n_synapses: int, number of synapses in total 
			n_bias: int, number of bias terms (must be coherent with n_synapses)
			hnode: list, list containing the number of nodes (hidden).
			weightmethod: str, type of weight initialization method (allowed (None, xavier, normxavier)
			weightlim: tuple, of two argument (min and max range for cromosome initialization)
			bias_weightlim: tuple, f two argument (min and max range for bias cromosome initialization)
			cross_prob: float, crossover probability
			swap_prob: float, (if crossover cross_strategy = uniform) swap probability between parents
			cross_strategy: bool, (allowed: "average", "uniform")
			mut_prob: float, mutation probability
			mut_sigma: float, mutation noise (e ~ N(0, mut_sigma))
			mut_decay: bool, whether to decay sigma over generations
			mut_decay_factor: floar, scaling factor:  $mut_sigma := mut_sigma × (mut_decay_factor) ^ generation
			perm_prob: float, coevolution permutation probability (to be applied
				over parents only according to the fitness)
			perm_adaptive: bool, adaptive permutation
		"""

		# initial check
		assert len(weightlim)==2, "Weights limits exceed: required (min, max)"
		assert weightmethod in (None, "normxavier", "unixavier")
		assert cross_strategy in ("average", "uniform"),  "Function not implemented! allowed: (\"average\", \"uniform\")"

		# store general parameter
		self.popsize = popsize
		self.n_synapses = n_synapses
		self.n_bias = n_bias
		self.generation = 0
		self.weightlim = weightlim
		minW, maxW = self.weightlim
		if minW > maxW:
			minW, maxW = maxW, minW
		self.bias_weightlim = bias_weightlim
		minB, maxB = self.weightlim
		if minB > maxB:
			minB, maxB = maxB, minB

		# check hnode and n_synapses
		self.hnode = hnode.copy()
		self.hnode.insert(0, 64)
		self.hnode.insert(len(self.hnode), 1)
		tot_syn = 0
		for i in range(1,len(self.hnode)):
			tot_syn += self.hnode[i-1]*self.hnode[i]
		assert tot_syn == self.n_synapses, "Difference mismatch in n_synapses and hnode list!"

		# store hyperparams

		self.weightmethod = weightmethod
		self.cross_prob = cross_prob
		self.swap_prob = swap_prob
		self.cross_strategy = cross_strategy 
		self.mut_prob = mut_prob 
		self.mut_sigma = mut_sigma
		self.mut_decay = mut_decay
		self.mut_decay_factor = mut_decay_factor
		self.perm_prob = perm_prob
		self.perm_adaptive = perm_adaptive

		# initialize the matrix structure (weights)
		if self.weightmethod == None:
			self.matrix = np.random.uniform(minW, maxW, size=(self.n_synapses, self.popsize))
			self.bias = np.random.uniform(minB, maxB, size=(self.n_bias, self.popsize))

		# initialize the matrix structure considering Xavier/normalised Xavier 
		else:
			
			to_concat_syn, to_concat_bias = [], []

			for i in range(1,len(self.hnode)):
				n, m = self.hnode[i-1], self.hnode[i]

				if self.weightmethod == "normxavier":
					syn = np.random.normal(0, (2/np.sqrt(n+m)), size = (n*m, self.popsize))
					bias = np.random.normal(0, (2/np.sqrt(n+m)), size = (m, self.popsize))

				elif self.weightmethod == "unixavier":
					syn = np.random.uniform(
						-(np.sqrt(6)/np.sqrt(n+m)), (np.sqrt(6)/np.sqrt(n+m)), size = (n*m, self.popsize)
						)
					bias = np.random.uniform(
						-(np.sqrt(6)/np.sqrt(n+m)), (np.sqrt(6)/np.sqrt(n+m)), size = (m, self.popsize)
						)

				to_concat_syn.append(syn)	
				to_concat_bias.append(bias)
					
			self.matrix = np.concatenate(to_concat_syn, axis = 0)
			self.bias = np.concatenate(to_concat_bias[0:-1], axis = 0) # excluding the bias term of the output node


		# initialize offspring
		self.offsprings = np.zeros((self.n_synapses, self.popsize//2))
		self.n_offsprings = self.popsize//2
		self.bias_offsprings = np.zeros((self.n_bias, self.popsize//2))


		# store all past fitness matrices
		self.past_generations = {0:self.matrix}
		self.past_fitness = dict()
		self.fitness = None

	def reshufle_order(self, randomidx:np.array):
		"""
		Once evolved, reshufle the elements to be tested in the tournament
		"""
		print("Evolution: reshufle players order for tournament")
		self.matrix = self.matrix[:, randomidx]
		self.bias = self.bias[:, randomidx]


	def update_fitness(self, new_fitness:np.array):
		"""
		Update the fitness vector
		Args:
			new_fitness: (list) vector of the new fitness
		"""
		print("Evolution: update fitness")

		assert new_fitness.shape[0] == self.popsize, "Fitness dimension mismatch!"
		if self.past_fitness == {}:
			self.past_fitness[0] = new_fitness
		else:
			self.past_fitness[self.generation] = new_fitness

		self.fitness = new_fitness


	def _selection(self, debug:bool=True):
		"""
		Select the n/2 virtual players among the current generation according to 
		the fitness function (score).
		The parent genotype matrix will be reduced by n//2
		"""
		# consider the idea to perform the computation in pandas !
		print("Evolution: selection")

		if debug:
			t0=time.time()
		tmp = [(i, self.fitness[i]) for i in range(len(self.fitness))]
		tmp.sort(key=lambda x: x[1], reverse=True)
		indexes = [tmp[i][0] for i in range(len(tmp))]
		self.fitness = np.array([tmp[i][1] for i in range(len(tmp))][0:self.popsize//2])

		# reorder the genome matrix according to fitness and extract the n/2 player
		self.matrix = self.matrix[:,indexes][:, 0:self.popsize//2]
		self.bias = self.bias[:, indexes][:, 0:self.popsize//2]

		if debug:
			t1=time.time()
			print("Selection: time elapsed", t1-t0)


	def _crossover(self):
		"""
		Apply crossover to the current genotype matrix (and store in the offsprings matrix):
		- given a certain probability (cross_prob):
		- combine elements of the genome according to a certain function
			to be defined
		Args used (to be not specified):
			cross_prob: (float) mutation probability
			cross_strategy: (str) crossover function, allowed: ("aritmetic", "uniform")
		"""
		print("Evolution: crossover")

		# gather a sufficient amount of parents to reproduce
		cp = self.cross_prob		
		while True:
			where_to_crossover = np.random.uniform(0, 1, size = (self.matrix.shape[1])) < cp
			if where_to_crossover.sum()<3: # and where_to_crossover.shape[0]<self.matrix.shape[1]//10:
				cp += 0.05
			else:
				break

		# extract the indexes of the parents that will reproduce
		indexes_parents = list(np.arange(0, self.matrix.shape[1])[where_to_crossover])
		indexes_genome  = list(np.arange(0, self.n_synapses))

		j = 0

		if self.cross_strategy=="average":

			while j < self.n_offsprings:
				# draw randomly choosen parents
				parent1, parent2 = random.sample(indexes_parents, 2)

				# add offsprings
				self.offsprings[:,j] = np.array([self.matrix[:, parent1], self.matrix[:, parent2]]).mean(axis=0)
				self.bias_offsprings[:,j] = np.array([self.bias[:, parent1], self.bias[:, parent2]]).mean(axis=0)
				j += 1

		elif self.cross_strategy=="uniform":

			while j < self.n_offsprings:
				# draw randomly choosen parents
				parent1, parent2 = random.sample(indexes_parents, 2)
				where_to_swap = np.random.uniform(0, 1, size = (len(indexes_genome))) < self.swap_prob
				where_to_swap_bias = np.random.uniform(0, 1, size = self.n_bias) < self.swap_prob

				# add offsprings
				self.offsprings[:, j]     = np.where(where_to_swap, self.matrix[:, parent2], self.matrix[:, parent1])
				self.offsprings[:, (j+1)] = np.where(where_to_swap, self.matrix[:, parent1], self.matrix[:, parent2])
				self.bias_offsprings[:, j]     = np.where(where_to_swap_bias, self.bias[:, parent2], self.bias[:, parent1])
				self.bias_offsprings[:, (j+1)] = np.where(where_to_swap_bias, self.bias[:, parent1], self.bias[:, parent2])
				j += 2


	def _mutation(self):
		"""
		Apply mutation to the current offsprings genotype matrix:
		for each sub-genome in the population (column) apply the following:
		- given a certain probability (mut_prob):
		- add to the current element of the genotype e ~ N(0, mut_sigma)
		Args (to be not specified):
			mut_prob: (float) mutation probability
			mut_sigma: (float) random variation
		"""
		print("Evolution: mutation")

		if self.mut_decay:
			sigma = self.mut_sigma * ((self.mut_decay_factor)**(self.generation-1))
		else:
			sigma = self.mut_sigma

		# initialize mutation and mut marks for synapses' offsprings
		mutation = np.random.normal(0, sigma, size = (self.offsprings.shape))
		where_to_mutate = np.random.uniform(0, 1, size = (self.offsprings.shape)) < self.mut_prob
		self.offsprings = self.offsprings + np.where(where_to_mutate, mutation, 0)

		# initialize mutation and mut marks for the bias matrix's offsrpings
		mutation_bias = np.random.normal(0, sigma, size = (self.bias_offsprings.shape))
		where_to_mutate_bias = np.random.uniform(0, 1, size = (self.bias_offsprings.shape)) < self.mut_prob
		self.bias_offsprings = self.bias_offsprings + np.where(where_to_mutate_bias, mutation_bias, 0)



	def _permute(self):
		"""
		Permute the genotype matrix (parents from the current generation) 
		within the same synaptic sub-population (neglecting the offspring matrix) 
		as in Gomez et al., 2008.
		This step needs to be applied after crossover+mutation.
		Args (to be not specified):
			perm_prob: (float) probability of resampling withing the same synapsis
		"""
		print("Evolution: permutation")

		indexes = np.arange(0,self.matrix.shape[1])

		synapses_bias = [self.matrix, self.bias]

		for i, mat in enumerate(synapses_bias):

			# condition in which a simple random permutation is applied
			if not self.perm_adaptive:
				where_to_shuffle = np.random.uniform(0, 1, size = (mat.shape)) < self.perm_prob
				
				for i in range(0, mat.shape[0]):
					curr_row, curr_where_to_shuffle = mat[i,:], where_to_shuffle[i,:]

					# if there are not enough elements to permute in the current row:
					if curr_where_to_shuffle.sum() <= 1: continue

					# shuffle synapses 
					to_shuffle, to_shuffle_idx = curr_row[curr_where_to_shuffle], indexes[curr_where_to_shuffle]
					np.random.shuffle(to_shuffle)
					mat[i,:][to_shuffle_idx] = to_shuffle

			# adaptive permutation: implementation from Gomez et al., 2008
			# it had been specified that the permutation is random although the image seems
			# to describe a higher fitness to lower fitness exchange.
			else:
				fmax, fmin = max(self.fitness), min(self.fitness)
				estimated_prob = 1 - ((np.array(self.fitness)-fmin)/(fmax-fmin))**(1/2)
				outcomes = np.random.uniform(0, 1, size = mat.shape)
				permutation_mark = outcomes < estimated_prob

				for i in range(0, mat.shape[0]):
					curr_row, curr_where_to_shuffle = mat[i,:], permutation_mark[i,:]
					to_shuffle, to_shuffle_idx = curr_row[curr_where_to_shuffle], indexes[curr_where_to_shuffle]
					np.random.shuffle(to_shuffle)
					mat[i,:][to_shuffle_idx] = to_shuffle

			if i == 0:
				self.matrix = mat
			else:
				self.bias = mat


	def _append_matrices(self):
		"""
		Unify the previous generation parent genotype with the current offsprings
		"""
		print("Evolution: appending matrices")
		self.matrix = np.append(self.matrix, self.offsprings, axis = 1)
		self.bias = np.append(self.bias, self.bias_offsprings, axis = 1)


	def evolve(self, new_fitness):
		"""
		Apply mutation, crossover to generate offsprings and parent permutation 
		"""
		self.update_fitness(new_fitness)
		self._selection()
		self._crossover()
		self._mutation()
		self._permute()
		self._append_matrices()
		self.past_generations[self.generation] = self.matrix
		self.generation += 1
		print("Evolution: Generation ", self.generation, " completed")


	def save_progress(self, outdir:str):
		"""
		Save current progress in a npy file
		"""
		print("Evolution: saving synapses and bias mat")
		np.save(f"{outdir}_g{'0'+str(self.generation) if len(str(self.generation))==1 else str(self.generation)}_synapses", self.matrix)
		np.save(f"{outdir}_g{'0'+str(self.generation) if len(str(self.generation))==1 else str(self.generation)}_bias", self.bias)


	def load_matrix(self, filename:str):
		"""
		Load matrix of synapses from previously completed job:
		check the generation number and update it
		"""
		loaded_mat = np.load(filename)
		assert self.matrix.shape == loaded_mat.shape, "Matrix dimension mismatch!"

		splitted = filename.split("/")[-1].split("_")
		for i in range(len(splitted)):
			if splitted[i].startswith("g"):
				generation = int(splitted[i][1:])
				break

		print(f"Loading synapses matrix:\n dimension: {loaded_mat.shape}  from {filename}")
		self.matrix = loaded_mat
		self.generation = generation


	def load_bias(self, filename:str):
		"""
		Load matrix of biases from previously completed job:
		check the generation number and update it
		"""
		loaded_mat = np.load(filename)
		assert self.bias.shape == loaded_mat.shape, "Matrix dimension mismatch!"

		splitted = filename.split("/")[-1].split("_")
		for i in range(len(splitted)):
			if splitted[i].startswith("g"):
				generation = int(splitted[i][1:])
				break

		print(f"Loading synapses matrix:\n dimension: {loaded_mat.shape}  from: {filename}")
		self.bias = loaded_mat
		self.generation = generation


	def __str__(self):
		return f"""
				crossprob:{self.cross_prob}_swapprob:{self.swap_prob}_crossstragegy:{self.cross_strategy}_mutprob:{self.mut_prob}_mutsigma:{self.mut_sigma}_mutdecay:{self.mut_decay}_mutdecayfactor:{self.mut_decay_factor}_permprob:{self.perm_prob}_permadapt:{self.perm_adaptive}
				"""
