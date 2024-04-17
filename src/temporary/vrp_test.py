import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from database.database_connection import DatabaseConnection

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

    db = ConDB()

    hotel = db.get_hotel_node(idhotel)
    tur = db.get_tour_nodes(tourid)
    timematrix = db.get_time_matrix(hotel._id, tourid)

    return hotel, tur, timematrix