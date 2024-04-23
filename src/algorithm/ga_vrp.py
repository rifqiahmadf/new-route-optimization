import sys
import os
import random
import time

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
        selected_individuals = []
        for _ in range(len(individuals)):
            competitors = random.choices(
                list(zip(individuals, fitness_values)), k=tournament_size
            )
            winner = max(competitors, key=lambda competitor: competitor[1])[0]
            selected_individuals.append(winner)
        return selected_individuals

    def roulette_wheel_selection(self, individuals, fitness_values):
        """
        Perform roulette wheel selection from a population.

        This method selects individuals based on their fitness proportionate to the total
        fitness of the population, simulating a roulette wheel mechanism.

        Parameters:
            individuals (list): The current population of individuals.
            fitness_values (list): The fitness values corresponding to the individuals.

        Returns:
            list: The selected subset of the population.
        """
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]
        selected_individuals = random.choices(
            individuals, weights=probabilities, k=len(individuals)
        )
        return selected_individuals

    def construct_solution(self):
        start_time = time.time()
        """
        Construct a solution for the Genetic Algorithm in Vehicle Routing Problem.

        This method initializes a population of random solutions and iteratively
        applies crossover and mutation operators to create new offsprings. It then
        evaluates the fitness of the combined population and selects the best
        individuals for the next generation. The process repeats for a maximum number
        of iterations or until an idempotent state is reached where no further
        improvements are observed.

        Returns:
            tuple: A dictionary representing the best solution found and its fitness.
        """
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

            use_tournament_selection = True

            if use_tournament_selection:
                population = self.tournament_selection(all_individuals, fitness_values)
            else:
                population = self.roulette_wheel_selection(
                    all_individuals, fitness_values
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
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(execution_time)
        return solution_dict, best_fitness

    def test(self):

        population = [random.sample(self.tour, len(self.tour)) for _ in range(100)]
        start_time = time.time()  # Record start time
        fitness_values = [
            self.calculate_maut(
                self.convert_solution_list_to_dict(self.split_itinerary(individual))
            )
            for individual in population
        ]
        self.roulette_wheel_selection(population, fitness_values)
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        print(execution_time)
