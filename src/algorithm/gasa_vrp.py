import sys
import os
import random
import time
import numpy as np

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from algorithm.vrp_framework import VRPFramework


class GASAVRP(VRPFramework):
    def __init__(
        self,
        population_size=50,
        crossover_rate=0.8,
        mutation_rate=0.2,
        max_iter=300,
        max_idem=15,
        temperature=15000,
        cooling_rate=0.99,
        stopping_temperature=0.0002,
        random_state=None,
    ):
        super().__init__()

        # parameter setting
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.max_idem = max_idem

        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature

        random.seed(random_state)

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

    def swap_mutation(self, individual):
        """
        Perform a swap mutation on a given individual.

        A swap mutation chooses two positions at random in the individual's
        sequence and swaps their values. This is done to introduce variation
        in the genetic algorithm population.

        Parameters:
        individual (list): The individual from the population to mutate.
                        An individual is typically represented as a list.

        Returns:
        list: A new individual with two elements swapped, which introduces
            a small amount of variation.
        """
        # Create a shallow copy of the individual to avoid modifying the original
        mutated_individual = individual[:]

        # Randomly select two positions to swap
        pos1, pos2 = random.sample(range(len(mutated_individual)), 2)

        # Perform the swap mutation
        mutated_individual[pos1], mutated_individual[pos2] = (
            mutated_individual[pos2],
            mutated_individual[pos1],
        )

        return mutated_individual

    def tournament_selection(self, individuals, fitness_values, tournament_size=3):
        """
        Perform tournament selection from a population.

        This method randomly picks 'tournament_size' individuals from the population and
        selects the best one based on fitness values.

        Parameters:
            individuals (list): The current population of individuals.
            fitness_values (list): The fitness values corresponding to the individuals.
            tournament_size (int): The number of individuals to compete in each tournament.

        Returns:
            list: The selected subset of the population.
        """
        individuals = np.array(individuals)
        fitness_values = np.array(fitness_values)

        selected_individuals = []
        for _ in range(len(individuals)):
            competitors_indices = np.random.choice(
                len(individuals), size=tournament_size, replace=False
            )
            competitors_fitness = fitness_values[competitors_indices]
            winner_index = np.argmax(competitors_fitness)
            winner = individuals[competitors_indices[winner_index]]
            selected_individuals.append(winner)
        return selected_individuals

    def roulette_wheel_selection(
        self, all_individuals, fitness_values, population_size
    ):
        """
        Select a population from all individuals based on their fitness values.

        This function randomly selects individuals from the population with probabilities
        proportional to their fitness values.

        Parameters:
            all_individuals (list): List of all individuals.
            fitness_values (list): Fitness values corresponding to the individuals.
            population_size (int): Size of the population to select.

        Returns:
            list: The selected population.
        """
        population = random.choices(
            all_individuals,
            weights=np.array(fitness_values) / sum(fitness_values),
            k=self.population_size,
        )
        return population

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

    def construct_solution(self):
        solution_dict = {}
        best_fitness = 0
        idem_counter = 0

        # Initialize the population with random samples of tours
        population = [
            random.sample(self.tour, len(self.tour))
            for _ in range(self.population_size)
        ]

        for _ in range(self.max_iter):
            offspring = []

            # Crossover and mutation processes
            for ind in range(0, self.population_size, 2):
                parent1, parent2 = population[ind], population[ind + 1]

                # Apply crossover based on the crossover rate
                if random.uniform(0, 1) < self.crossover_rate:
                    child1, child2 = self.partially_mapped_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]

                # Apply mutation based on the mutation rate
                if random.uniform(0, 1) < self.mutation_rate:
                    child1 = self.swap_mutation(child1)
                if random.uniform(0, 1) < self.mutation_rate:
                    child2 = self.swap_mutation(child2)

                offspring.extend([child1, child2])

            # Selection process
            all_individuals = population + offspring
            fitness_values = [
                self.calculate_maut(
                    self.convert_solution_list_to_dict(self.split_itinerary(individual))
                )
                for individual in all_individuals
            ]

            use_tournament_selection = False

            if use_tournament_selection:
                population = self.tournament_selection(all_individuals, fitness_values)
            else:
                population = self.roulette_wheel_selection(
                    all_individuals=all_individuals,
                    fitness_values=fitness_values,
                    population_size=self.population_size,
                )

            # Determine the best solution in the current generation
            best_solution = max(
                population,
                key=lambda x: self.calculate_maut(
                    self.convert_solution_list_to_dict(self.split_itinerary(x))
                ),
            )
            current_fitness = self.calculate_maut(
                self.convert_solution_list_to_dict(self.split_itinerary(best_solution))
            )

            # Update the best solution found so far
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                solution_dict = self.convert_solution_list_to_dict(
                    self.split_itinerary(best_solution)
                )
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    break

            solution_dict = self.convert_solution_list_to_dict(
                self.split_itinerary(best_solution)
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
