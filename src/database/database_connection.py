import pymysql as mdb
import datetime

import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from database.travel_node import TravelNode

class DatabaseConnection:
    """
    A class to handle database connections and queries for a tourism recommendation system.

    Attributes:
        connection (pymysql.connections.Connection): A connection to the MySQL database.
    """

    def __init__(self):
        """
        Initializes the database connection.
        """
        self.connection = self.connect()

    def connect(self):
        """
        Establishes a connection to the MySQL database.

        Returns:
            pymysql.connections.Connection: The database connection object.
        """
        return mdb.connect(
            host="127.0.0.1",
            user="root",
            password="",
            db="rekomendasi_wisata",
            port=3306,
            charset="utf8mb4",
            cursorclass=mdb.cursors.DictCursor,
        )

    def select_all_from_table(self, table):
        """
        Fetches all records from a specified table.

        Parameters:
            table (str): The name of the table from which to fetch data.

        Returns:
            list: A list of dictionaries representing each row from the table.
        """
        with self.connection.cursor() as cursor:
            sql = f"SELECT * FROM {table}"
            cursor.execute(sql)
            return cursor.fetchall()

    def fetch_schedule(self, place_id, day="minggu"):
        """
        Retrieves the opening and closing times for a given place on a specified day.

        Parameters:
            place_id (int): The unique identifier of the place.
            day (str): Day in Indonesia.

        Returns:
            tuple: Opening and closing times as datetime.time objects.
        """
        with self.connection.cursor() as cursor:
            sql = """
            SELECT pj_jam_buka, pj_jam_tutup
            FROM posts_jadwal
            WHERE pj_id_tempat = %s AND pj_hari = %s
            """
            cursor.execute(sql, (place_id, day))
            schedule = cursor.fetchone()
            return schedule["pj_jam_buka"], schedule["pj_jam_tutup"]

    def fetch_tour_nodes(self, place_ids):
        """
        Retrieves tour node information for a list of place IDs.

        Parameters:
            place_ids (list[int]): A list of place IDs.

        Returns:
            list[TravelNode]: A list of TravelNode objects for the tour.
        """
        with self.connection.cursor() as cursor:
            in_p = ", ".join(["%s"] * len(place_ids))
            sql = f"""
            SELECT post_id, post_title_id, post_lat, post_long, post_rating, post_type, post_kunjungan_sec, post_tarif
            FROM posts
            WHERE post_id IN ({in_p})
            """
            cursor.execute(sql, place_ids)
            places = cursor.fetchall()
            tour = []
            for place in places:
                open_time, close_time = self.fetch_schedule(place["post_id"])
                open_time = datetime.time(
                    open_time.seconds // 3600, (open_time.seconds // 60) % 60, 0
                )
                close_time = datetime.time(
                    close_time.seconds // 3600, (close_time.seconds // 60) % 60, 0
                )
                node = TravelNode(
                    place["post_id"],
                    place["post_title_id"],
                    place["post_lat"],
                    place["post_long"],
                    place["post_kunjungan_sec"],
                    place["post_rating"],
                    place["post_tarif"],
                    place["post_type"],
                    open_time,
                    close_time,
                )
                tour.append(node)
            return tour

    def fetch_hotel_node(self, hotel_id):
        """
        Retrieves hotel node information for a specified hotel ID.

        Parameters:
            hotel_id (int): The unique identifier of the hotel.

        Returns:
            TravelNode: A TravelNode object for the hotel.
        """
        open_time, close_time = datetime.time(0, 0), datetime.time(23, 59)
        visit_duration = 0

        with self.connection.cursor() as cursor:
            sql = """
            SELECT post_id, post_title_id, post_lat, post_long, post_rating, post_type, post_tarif
            FROM posts
            WHERE post_id = %s
            """
            cursor.execute(sql, (hotel_id,))
            hotel = cursor.fetchone()
            return TravelNode(
                hotel["post_id"],
                hotel["post_title_id"],
                hotel["post_lat"],
                hotel["post_long"],
                visit_duration,
                hotel["post_rating"],
                hotel["post_tarif"],
                hotel["post_type"],
                open_time,
                close_time,
            )

    def fetch_time_matrix(self, hotel_id, place_ids):
        """
        Constructs a time matrix for travel between places and a hotel.

        Parameters:
            hotel_id (int): The unique identifier of the hotel.
            place_ids (list[int]): A list of place IDs.

        Returns:
            dict: A nested dictionary representing the time matrix.
        """
        with self.connection.cursor() as cursor:
            ids = tuple(place_ids + [hotel_id])
            sql = """
            SELECT pt_id, pt_a, pt_b, pt_waktu
            FROM posts_timematrix
            WHERE pt_a IN %s AND pt_b IN %s
            """
            cursor.execute(sql, (ids, ids))
            matrix = cursor.fetchall()
            time_matrix = {}
            for entry in matrix:
                if entry["pt_a"] not in time_matrix:
                    time_matrix[entry["pt_a"]] = {}
                time_matrix[entry["pt_a"]][entry["pt_b"]] = {
                    "duration": entry["pt_waktu"]
                }
            return time_matrix
