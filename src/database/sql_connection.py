import pandas as pd
import pymysql

# Database connection settings
host = "127.0.0.1"
user = "root"
passwd = ""
db_name = "rekomendasi_wisata"

def get_db_connection():
    """
    Establishes a connection to the MySQL database.

    Returns:
        pymysql.connections.Connection: The database connection object.
    """
    return pymysql.connect(host=host, user=user, passwd=passwd, db=db_name, port=3306)

def read_from_sql(query, connection):
    """
    Executes a SQL query and returns the result as a Pandas DataFrame.

    Parameters:
        query (str): The SQL query to execute.
        connection (pymysql.connections.Connection): The database connection object.

    Returns:
        pd.DataFrame: The result of the SQL query as a Pandas DataFrame.
    """
    try:
        return pd.read_sql(query, connection)
    except Exception as e:
        raise e

def execute_sql(query, connection):
    """
    Executes a SQL command in the database.

    Parameters:
        query (str): The SQL command to execute.
        connection (pymysql.connections.Connection): The database connection object.

    Returns:
        str: A message indicating successful execution.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
        connection.commit()
        return "Successfully executed."
    except Exception as e:
        raise e

def write_df_to_sql(df, table_name, if_exists='append', dtype=None, connection=None):
    """
    Writes a Pandas DataFrame to a SQL table without using SQLAlchemy and without dropping the table.

    Parameters:
        df (pd.DataFrame): The DataFrame to write.
        table_name (str): The name of the target SQL table.
        if_exists (str): How to behave if the table already exists ('fail', 'append', 'ignore').
        dtype (dict, optional): Data types to force for columns.
        connection (pymysql.connections.Connection, optional): The database connection object.

    Returns:
        str: A message indicating successful writing operation.
    """
    # Get a database connection if not already provided
    if connection is None:
        connection = get_db_connection()
    
    try:
        with connection.cursor() as cursor:
            if if_exists == 'fail':
                cursor.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                if cursor.fetchone():
                    return f"Table {table_name} already exists."

            # Prepare the SQL for inserting data
            cols = ','.join([f"`{col}`" for col in df.columns])
            val_placeholders = ','.join(['%s'] * len(df.columns))
            insert_query = f"INSERT INTO {table_name} ({cols}) VALUES ({val_placeholders})"
            
            # Execute the insert query for each row in the DataFrame
            for row in df.itertuples(index=False, name=None):
                cursor.execute(insert_query, row)
            
            connection.commit()
            return f"Successfully written to {db_name} as {table_name}"
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        if connection is not None:
            connection.close()