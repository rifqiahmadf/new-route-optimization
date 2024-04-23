import copy
import datetime
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from database.database_connection import DatabaseConnection


class VRPFramework:
    """
    Represents a Vehicle Routing Problem (VRP) framework tailored for tourism applications. This class handles the
    setup and execution of routing algorithms, considering various factors such as cost, duration, and rating,
    using Multi-Attribute Utility Theory (MAUT) for decision-making.

    Attributes:
        db (DatabaseConnection): Connection to the database to fetch necessary data.
        initial_solution (list): A list to store an initial solution setup, if any.
        remaining_nodes (list): Nodes that have not yet been visited or decided upon.
        tour (list): List of all nodes representing the tour spots.
        hotel (object): Node representing the hotel/start/end point.
        time_matrix (dict): Matrix that contains travel time between any two nodes.
        maximum_travel_duration (int): The maximum duration for travel between nodes.
        travel_days (int): The total number of travel days in the itinerary.
        degree_duration (float): Weight for the duration in the MAUT calculation.
        degree_cost (float): Weight for the cost in the MAUT calculation.
        degree_rating (float): Weight for the rating in the MAUT calculation.
        degree_point_of_interest (float): Weight for the point of interest factor in MAUT.
        degree_penalty_point_of_interest (float): Penalty weight for missing points of interest in MAUT.
        degree_penalty_time (float): Penalty weight for time deviations in MAUT.
        minimum_rating (float): Minimum rating across all nodes for scaling purposes.
        maximum_rating (float): Maximum rating across all nodes for scaling purposes.
        minimum_cost (float): Minimum cost across all nodes for scaling purposes.
        maximum_cost (float): Maximum cost across all nodes for scaling purposes.
        minimum_duration (int): Minimum travel duration across all nodes for scaling purposes.
        maximum_duration (int): Maximum travel duration across all nodes for scaling purposes.
        minimum_point_of_interest (int): Minimum count of points of interest for scaling.
        maximum_point_of_interest (int): Maximum count of points of interest for scaling.
        minimum_penalty_point_of_interest (int): Minimum penalty for missing points of interest.
        maximum_penalty_point_of_interest (int): Maximum penalty for missing points of interest.
        minimum_penalty_time (int): Minimum time penalty for deviations.
        maximum_penalty_time (int): Maximum time penalty for deviations.
    """

    def __init__(self):
        self.db = DatabaseConnection()

        # Initialize the solution and node tracking lists
        self.initial_solution = []
        self.remaining_nodes = []

        # Initialize the data model settings
        self.tour = None
        self.hotel = None
        self.time_matrix = None
        self.maximum_travel_duration = None
        self.travel_days = None

        # Initialize weights and degrees for MAUT calculation
        self.degree_duration = 1
        self.degree_cost = 1
        self.degree_rating = 1
        self.degree_point_of_interest = 1
        self.degree_penalty_point_of_interest = 1
        self.degree_penalty_time = 1

        # Initialize the scaling settings
        self.minimum_rating = None
        self.maximum_rating = None
        self.minimum_cost = None
        self.maximum_cost = None
        self.minimum_duration = None
        self.maximum_duration = None
        self.minimum_point_of_interest = None
        self.maximum_point_of_interest = None
        self.minimum_penalty_point_of_interest = None
        self.maximum_penalty_point_of_interest = None
        self.minimum_penalty_time = None
        self.maximum_penalty_time = None

    def set_model(
        self,
        tour,
        hotel,
        time_matrix,
        travel_days=3,
        departure_time=datetime.time(8, 0, 0),
        maximum_travel_time=datetime.time(20, 0, 0),
        degree_duration=1,
        degree_cost=1,
        degree_rating=1,
    ):
        """
        Configures the VRP model with a specific tour setup, including nodes, hotel, and travel time constraints.

        Parameters:
            tour (list): A list of node objects that represent the tour stops.
            hotel (Node): The node object representing the hotel/start/end point of the tour.
            time_matrix (dict): A dictionary representing the time taken to travel between each pair of nodes.
            travel_days (int, optional): The number of days over which the tour is spread. Defaults to 3.
            departure_time (datetime.time, optional): The daily start time for the tour. Defaults to 8:00 AM.
            maximum_travel_time (datetime.time, optional): The daily end time for the tour. Defaults to 8:00 PM.
            degree_duration (float, optional): The weight for duration in the MAUT calculation. Defaults to 1.
            degree_cost (float, optional): The weight for cost in the MAUT calculation. Defaults to 1.
            degree_rating (float, optional): The weight for rating in the MAUT calculation. Defaults to 1.

        Effects:
            Initializes the model with the specified parameters and calculates the scaling factors for the MAUT
            metrics based on the provided tour data.
        """
        # Deep copy tour and hotel to prevent modification of original data
        self.tour = copy.deepcopy(tour)
        self.hotel = copy.deepcopy(hotel)

        # Set travel days and time constraints
        self.travel_days = travel_days
        self.hotel.departure_time = departure_time
        self.maximum_travel_time = maximum_travel_time

        # Deep copy the time matrix to ensure that changes do not affect the original data
        self.time_matrix = copy.deepcopy(time_matrix)

        # Set MAUT calculation weights
        self.degree_duration = degree_duration
        self.degree_cost = degree_cost
        self.degree_rating = degree_rating

        # Extract ratings and costs from the tour nodes to calculate min/max values for scaling
        tour_ratings = [tour_node.rating for tour_node in self.tour]
        tour_costs = [tour_node.cost for tour_node in self.tour]

        # Calculate the minimum and maximum values for ratings and costs
        self.minimum_rating = min(tour_ratings)
        self.maximum_rating = max(tour_ratings)
        self.minimum_cost = min(tour_costs)
        self.maximum_cost = sum(tour_costs)

        # Calculate the range of possible durations using the specified departure and travel times
        self.minimum_duration = 0
        self.maximum_duration = (
            self.seconds_difference_between_times(departure_time, maximum_travel_time)
            * travel_days
        )

        # Point of interest metrics based solely on the number of nodes in the tour
        self.minimum_point_of_interest = 0
        self.maximum_point_of_interest = len(self.tour)
        self.minimum_penalty_point_of_interest = 0
        self.maximum_penalty_point_of_interest = len(self.tour)

        # Calculate the potential penalties for time based on the inverse of available tour time
        self.minimum_penalty_time = 0
        self.maximum_penalty_time = (
            24 * 3600
            - self.seconds_difference_between_times(maximum_travel_time, departure_time)
        ) * travel_days

    def set_maximum_iterations(self, maximum_iterations):
        """
        Sets the maximum number of iterations for the optimization algorithm.

        Parameters:
            maximum_iterations (int): The upper limit for the number of iterations to perform.
        """
        self.maximum_iterations = maximum_iterations

    def time_to_seconds(self, time):
        """
        Converts a datetime.time object to the total number of seconds past midnight.

        Parameters:
            time (datetime.time): The time to convert.

        Returns:
            int: Total seconds since midnight.
        """
        return (time.hour * 3600) + (time.minute * 60) + time.second

    def seconds_to_time(self, seconds):
        """
        Converts a number of seconds back into a datetime.time object.

        Parameters:
            seconds (int): The number of seconds to convert.

        Returns:
            datetime.time: The equivalent time object.
        """
        return datetime.time(seconds // 3600, (seconds // 60) % 60, seconds % 60)

    def seconds_difference_between_times(self, time_a, time_b):
        """
        Calculates the difference in seconds between two datetime.time objects.

        Parameters:
            time_a (datetime.time): The start time.
            time_b (datetime.time): The end time.

        Returns:
            int: The difference in seconds between time_a and time_b.
        """
        return self.time_to_seconds(time_b) - self.time_to_seconds(time_a)

    def min_max_scaler(self, minimum_value, maximum_value, value):
        """
        Scales a given value between 0 and 1 based on the minimum and maximum possible values.

        Parameters:
            minimum_value (float): The minimum value the variable can take.
            maximum_value (float): The maximum value the variable can take.
            value (float): The actual value to scale.

        Returns:
            float: A normalized value between 0 and 1.
        """
        # Return 0 to avoid division by zero if min and max values are the same
        return (
            0
            if maximum_value == minimum_value
            else (value - minimum_value) / (maximum_value - minimum_value)
        )

    def calculate_maut(
        self, solutions, consider_total_point_of_interest=True, use_penalty=True
    ):
        """
        Calculates the Multi-Attribute Utility Theory (MAUT) score for a given set of solutions. This score
        helps to evaluate the overall utility of different tour configurations based on various attributes.

        Parameters:
            solutions (list of dicts): Each dictionary represents a day's itinerary with indices, ratings, costs, and durations.
            consider_total_point_of_interest (bool): Determines if the total number of points of interest should affect the MAUT score.
            use_penalty (bool): Determines if penalties for missing points of interest and time deviations should be applied.

        Returns:
            float: The computed MAUT score, a weighted average of various scores derived from the itinerary attributes.
        """

        # Aggregating indices, ratings, and costs across all days
        index_list = sum([i["index"] for i in solutions], [])
        rating_list = sum([i["ratings"] for i in solutions], [])
        cost_list = sum([i["costs"] for i in solutions], [])

        # Aggregate durations for each day
        duration_list = [i["times"] for i in solutions]

        # Calculate average rating and then normalize and weight it
        average_rating = sum(rating_list) / len(rating_list)
        score_rating = (
            self.min_max_scaler(
                self.minimum_rating, self.maximum_rating, average_rating
            )
            * self.degree_rating
        )

        # Calculate total cost and then normalize and weight it
        total_cost = sum(cost_list)
        score_cost = (
            1 - self.min_max_scaler(self.minimum_cost, self.maximum_cost, total_cost)
        ) * self.degree_cost

        # Calculate total duration and normalize and weight it
        duration_per_day = [
            self.seconds_difference_between_times(i[0], i[-1]) for i in duration_list
        ]
        total_duration = sum(duration_per_day)
        score_duration = (
            1
            - self.min_max_scaler(
                self.minimum_duration, self.maximum_duration, total_duration
            )
        ) * self.degree_duration

        # Calculate and weight the point of interest score if enabled
        count_point_of_interest = len(index_list)
        score_point_of_interest = (
            self.min_max_scaler(
                self.minimum_point_of_interest,
                self.maximum_point_of_interest,
                count_point_of_interest,
            )
            if consider_total_point_of_interest
            else 0
        )

        score_penalty_point_of_interest = 0
        score_penalty_time = 0

        if use_penalty:
            # Calculate penalty for unvisited nodes
            penalty_index = [
                node.node_id for node in self.tour if node.node_id not in index_list
            ]
            count_penalty = len(penalty_index)
            score_penalty_point_of_interest = (
                1
                - self.min_max_scaler(
                    self.minimum_penalty_point_of_interest,
                    self.maximum_penalty_point_of_interest,
                    count_penalty,
                )
            ) * self.degree_penalty_point_of_interest

            # TODO FIX DESCRIPTION: early routes
            # Calculate time penalties for exceeding the maximum travel time
            penalty_per_day = [
                max(
                    self.seconds_difference_between_times(
                        i[-1], self.maximum_travel_time
                    ),
                    0,
                )
                for i in duration_list
            ]
            total_time_penalty = sum(penalty_per_day)
            score_penalty_time = (
                1
                - self.min_max_scaler(
                    self.minimum_penalty_time,
                    self.maximum_penalty_time,
                    total_time_penalty,
                )
            ) * self.degree_penalty_time

        # Calculate the overall MAUT score as a weighted sum of all component scores
        numerator = (
            score_rating
            + score_cost
            + score_duration
            + score_point_of_interest
            + score_penalty_point_of_interest
            + score_penalty_time
        )
        denominator = (
            self.degree_rating
            + self.degree_cost
            + self.degree_duration
            + (self.degree_point_of_interest if consider_total_point_of_interest else 0)
            + (self.degree_penalty_point_of_interest if use_penalty else 0)
            + (self.degree_penalty_time if use_penalty else 0)
        )
        maut_score = numerator / denominator
        return maut_score

    # TODO BACA
    def maut_between_two_nodes(self, current_node, next_node):
        """
        Calculates the Multi-Attribute Utility Theory (MAUT) score between two nodes considering ratings, costs, and time.

        Parameters:
            current_node: The starting node in the comparison.
            next_node: The destination node in the comparison.

        Returns:
            float: The calculated MAUT score.
        """
        score_rating = self.degree_rating * self.min_max_scaler(
            self.minimum_rating, self.maximum_rating, next_node.rating
        )
        score_cost = self.degree_cost * (
            1
            - self.min_max_scaler(self.minimum_cost, self.maximum_cost, next_node.cost)
        )
        score_duration = self.degree_duration * (
            1
            - self.min_max_scaler(
                self.minimum_duration,
                self.maximum_duration,
                self.time_matrix[current_node.node_id][next_node.node_id]["duration"],
            )
        )
        maut_score = (score_rating + score_cost + score_duration) / (
            self.degree_rating + self.degree_cost + self.degree_duration
        )
        return maut_score

    def is_next_node_feasible(self, current_node, next_node):
        """
        Checks whether moving to the next node is feasible within the operational and travel time constraints.

        Parameters:
            current_node: The node from which the move is initiated.
            next_node: The potential destination node.

        Returns:
            bool: True if the move is feasible within the given constraints, otherwise False.
        """
        travel_time_to_next_node = (
            self.time_to_seconds(current_node.departure_time)
            + self.time_matrix[current_node.node_id][next_node.node_id]["duration"]
            + next_node.visit_duration
        )
        return travel_time_to_next_node <= self.time_to_seconds(
            self.maximum_travel_time
        ) and travel_time_to_next_node <= self.time_to_seconds(next_node.close_time)

    def set_next_node_departure_arrival_time(self, current_node, next_node):
        """
        Sets the arrival and departure times for the next node based on travel and visit durations.

        Parameters:
            current_node: The node from which the journey is starting.
            next_node: The destination node where times are being set.

        Returns:
            next_node: The updated node with set arrival and departure times.
        """
        travel_time_to_next_node = self.time_matrix[current_node.node_id][
            next_node.node_id
        ]["duration"]
        arrival_time = (
            self.time_to_seconds(current_node.departure_time) + travel_time_to_next_node
        )
        # Ensure not arriving before opening time
        arrival_time = max(arrival_time, self.time_to_seconds(next_node.open_time))
        next_node.arrival_time = self.seconds_to_time(arrival_time)

        # Set departure time unless it's a hotel (hotels might not have a set visit duration)
        if next_node.node_type.lower() != "hotel":
            next_node.departure_time = self.seconds_to_time(
                arrival_time + next_node.visit_duration
            )
        return next_node

    def convert_solution_list_to_dict(self, solutions):
        """
        Converts a list of daily node visit solutions into a dictionary format that includes the node IDs,
        arrival times, ratings, and costs for each node each day.

        Parameters:
            solutions (list): A list of lists, where each inner list represents nodes visited in one day.

        Returns:
            list: A list of dictionaries with details for each day's node visits.
        """
        solution_dictionary = []
        for day_nodes in solutions:
            day_solution = {
                "index": [node.node_id for node in day_nodes],
                "times": [self.hotel.departure_time]
                + [node.arrival_time for node in day_nodes]
                + [
                    self.seconds_to_time(
                        self.time_to_seconds(day_nodes[-1].departure_time)
                        + self.time_matrix[day_nodes[-1].node_id][self.hotel.node_id][
                            "duration"
                        ]
                    )
                ],
                "ratings": [node.rating for node in day_nodes],
                "costs": [node.cost for node in day_nodes],
            }
            solution_dictionary.append(day_solution)
        return solution_dictionary

    def split_itinerary(self, initial_itinerary):
        """
        Distributes an initial itinerary list into a planned multi-day travel schedule based on node availability
        and travel constraints, ensuring each node is visited only once across the days.

        Parameters:
            initial_itinerary (list): A list of all possible nodes to be visited.

        Returns:
            list: A 2D list where each sublist represents a day's worth of node visits.
        """
        final_solution = []  # 2D list for storing each day's itinerary
        visited_nodes = set()  # Set to track visited node IDs to avoid revisits

        for day in range(1, self.travel_days + 1):
            current_node = self.hotel
            day_solution = []  # List to collect nodes for the current day
            # Filter candidates that have not been visited yet
            next_node_candidates = [
                node for node in initial_itinerary if node.node_id not in visited_nodes
            ]

            # Evaluate each candidate node to determine if it can be visited next
            for node in next_node_candidates:
                if self.is_next_node_feasible(current_node, node):
                    node = self.set_next_node_departure_arrival_time(current_node, node)
                    day_solution.append(node)
                    visited_nodes.add(node.node_id)
                    current_node = node

            # Add the day's solution to the final solution if any nodes were visited
            if day_solution:
                final_solution.append(day_solution)

            # Stop if all nodes have been visited
            if len(visited_nodes) == len(self.tour):
                break

        return final_solution
