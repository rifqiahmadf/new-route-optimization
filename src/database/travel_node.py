import datetime


class TravelNode:
    """
    Represents a node in a travel itinerary, such as a tourist destination or a hotel.

    Attributes:
        node_id (str): Unique identifier for the node.
        name (str): Name of the node.
        latitude (float): Latitude coordinate of the node.
        longitude (float): Longitude coordinate of the node.
        visit_duration (int): Time allocated for visiting the node, in seconds.
        rating (float): User rating of the node.
        cost (float): Cost associated with visiting the node.
        node_type (str): Type of the node, e.g., 'hotel' or 'location'.
        open_time (datetime.time): Opening time of the node.
        close_time (datetime.time): Closing time of the node.
        arrival_time (datetime.time): Calculated arrival time at the node.
        departure_time (datetime.time): Calculated departure time from the node.
    """

    DEFAULT_VISIT_DURATION = (
        3600  # Default visit duration in seconds for non-hotel nodes
    )

    def __init__(
        self,
        node_id,
        name,
        latitude,
        longitude,
        visit_duration,
        rating,
        cost,
        node_type,
        open_time,
        close_time,
    ):
        self.node_id = node_id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.node_type = node_type.lower()
        self.visit_duration = self.calculate_visit_duration(visit_duration)
        self.arrival_time = datetime.time(0, 0)
        self.departure_time = datetime.time(0, 0)
        self.rating = rating
        self.cost = cost
        self.open_time = open_time
        self.close_time = close_time

    def calculate_visit_duration(self, visit_duration):
        """
        Calculates the visit duration for the node. Uses a default visit duration for non-hotel nodes if no specific visit duration is provided.

        Parameters:
            visit_duration (int): The proposed duration of the visit in seconds.

        Returns:
            int: The visit time in seconds.
        """
        if int(visit_duration) != 0 and self.node_type != "hotel":
            return int(visit_duration)
        else:
            return self.DEFAULT_VISIT_DURATION

    def __repr__(self):
        """
        Represents the TravelNode instance as a string.

        Returns:
            str: String representation of the TravelNode instance.
        """
        # return f"({self.arrival_time} - {self.departure_time})"
        # return f"{self.name} ({self.arrival_time} - {self.departure_time})"
        return f"{self.node_id}"

