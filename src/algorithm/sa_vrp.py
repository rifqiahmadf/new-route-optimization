import sys
import os
import random
import time
import numpy as np

# Add parent directory to sys.path
current_directory = os.path.dirname(__file__)
parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
sys.path.append(parent_directory)

# Importing VRPFramework from the specified module
from algorithm.vrp_framework import VRPFramework


class SAVRP(VRPFramework):
    """
    Simulated Annealing for Vehicle Routing Problem (VRP).

    Attributes:
        temperature (float): Initial temperature.
        cooling_rate (float): Cooling rate for the temperature.
        stopping_temperature (float): Temperature at which the algorithm stops.
        random_state (int): Seed for random number generation.
    """

    def __init__(
        self,
        temperature=15000,
        cooling_rate=0.99,
        stopping_temperature=0.0002,
        random_state=None,
    ):
        """
        Initialize the SimulatedAnnealingVRP.

        Args:
            temperature (float): Initial temperature.
            cooling_rate (float): Cooling rate for the temperature.
            stopping_temperature (float): Temperature at which the algorithm stops.
            random_state (int): Seed for random number generation.
        """
        super().__init__()

        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature

        random.seed(random_state)

    def swap_operation(self, individual):
        """
        Perform a swap operation on the individual.

        Args:
            individual (list): List representing the individual.

        Returns:
            list: Mutated individual after performing the swap operation.
        """
        swapped_individual = individual[:]
        pos1, pos2 = random.sample(range(len(swapped_individual)), 2)
        swapped_individual[pos1], swapped_individual[pos2] = (
            swapped_individual[pos2],
            swapped_individual[pos1],
        )
        return swapped_individual

    def construct_solution(self):
        """
        Construct a solution using Simulated Annealing.

        Returns:
            tuple: A tuple containing the solution dictionary and its fitness value.
        """
        solution = random.sample(self.tour, len(self.tour))
        solution_dict = self.convert_solution_list_to_dict(
            self.split_itinerary(solution)
        )
        fitness = self.calculate_maut(solution_dict)

        while self.temperature >= self.stopping_temperature:
            # Generate a new solution
            new_solution = self.swap_operation(solution)
            new_fitness = self.calculate_maut(
                self.convert_solution_list_to_dict(self.split_itinerary(new_solution))
            )

            if new_fitness > fitness:
                solution = new_solution
                fitness = new_fitness
            else:
                # Calculate acceptance probability
                probability = np.exp(-(fitness - new_fitness) / self.temperature)
                if random.uniform(0, 1) < probability:
                    solution = new_solution
                    fitness = new_fitness

            self.temperature *= self.cooling_rate        

        solution_dict = self.convert_solution_list_to_dict(
            self.split_itinerary(solution)
        )
        fitness = self.calculate_maut(solution_dict)
        return solution_dict, fitness
