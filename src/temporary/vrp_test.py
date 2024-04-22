import sys
import os
import random

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from database.database_connection import DatabaseConnection
from database import sql_connection
from algorithm.ga_vrp import GAVRP


def generate_data(n=30, random_state=None):
    random.seed(random_state)
    query = """SELECT 
                    p.post_id,
                    p.post_type,
                    CASE
                        WHEN pj.pj_jam_buka = pj.pj_jam_tutup THEN "tutup"
                        ELSE "buka"
                    END AS is_operate
                FROM 
                    posts p
                LEFT JOIN
                    posts_jadwal pj
                    ON p.post_id = pj.pj_id_tempat AND pj.pj_hari = "minggu"
                """

    connection = sql_connection.get_db_connection()
    df_location = sql_connection.read_from_sql(query, connection)

    tourid = df_location[
        (df_location["post_type"] == "location")
        & (df_location["is_operate"] != "tutup")
    ]["post_id"].values.tolist()
    tourid = random.sample(tourid, n)
    idhotel = df_location[df_location["post_type"] == "hotel"][
        "post_id"
    ].values.tolist()
    idhotel = random.choice(idhotel)

    db = DatabaseConnection()

    hotel = db.fetch_hotel_node(idhotel)
    tur = db.fetch_tour_nodes(tourid)
    timematrix = db.fetch_time_matrix(idhotel, tourid)

    return hotel, tur, timematrix


random_state = 2024
hotel, tour, time_matrix = generate_data(random_state=random_state)

gavrp = GAVRP()
gavrp.set_model(
    tour=tour,
    hotel=hotel,
    time_matrix=time_matrix,
    travel_days=3,
    degree_cost=1,
    degree_duration=1,
    degree_rating=1,
)
gavrp.construct_solution()

# print(gavrp.tour)