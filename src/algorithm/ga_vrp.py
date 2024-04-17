import sys
import os
import random

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from algorithm.vrp_framework import VRPFramework


class GAVRP(VRPFramework):
    def __init__(
        self,
        population_size=50,
        crossover_rate=0.8,
        mutation_rate=0.2,
        max_iter=300,
        max_idem=15,
        random_state=None,
    ):
        super().__init__()

        # parameter setting
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.max_idem = max_idem

    def partially_mapped_crossover(self, parent1, parent2):  # PMX
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child1 = [None] * len(parent1)
        child2 = [None] * len(parent1)
        mapping1 = {parent1[i].node_id: parent2[i] for i in range(start, end + 1)}
        mapping2 = {parent2[i].node_id: parent1[i] for i in range(start, end + 1)}

        for i in range(len(parent1)):
            if start <= i <= end:
                child1[i], child2[i] = parent1[i], parent2[i]
            else:
                gene1, gene2 = parent2[i], parent1[i]
                while gene1.node_id in mapping1:
                    gene1 = mapping1[gene1.node_id]
                while gene2.node_id in mapping2:
                    gene2 = mapping2[gene2.node_id]
                child1[i], child2[i] = gene1, gene2
        return child1, child2

    def construct_solution(self):
        solution_dict = {}
        best_fitness = 0
        idem_counter = 0

        population = [random.sample(self.tour, len(self.tour)) for _ in range(2)]
        return VRPFramework.split_itinerary(self=self, initial_itinerary=population[0])
