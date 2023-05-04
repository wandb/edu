'''
Utils functions
'''

import json
import sqlite3
import snowflake.connector
from types import SimpleNamespace

class SQLConnector:
    """
    Class to handle Snowflake or SQLite database connections and queries
    """
    
    def __init__(self, config: SimpleNamespace, db_type: str):
        '''
        Takes a config object and a database type
        - db_type can be 'snowflake' or 'sqlite'
        '''
        self.config = config
        self.DATABASE_TYPE = db_type

    def connect_to_snowflake(self):
        '''
            Establish a connection to Snowflake
        '''
        conn = snowflake.connector.connect(
            account=self.config.SNOWFLAKE_ACCOUNT,
            user=self.config.SNOWFLAKE_USER,
            password=self.config.SNOWFLAKE_PASSWORD,
            # warehouse=config.SNOWFLAKE_WAREHOUSE,
            database=self.config.SNOWFLAKE_DATABASE,
            schema=self.config.SNOWFLAKE_SCHEMA
        )
        return conn
    
    def connect_to_sqlite(self):
        conn = sqlite3.connect(self.config.SQLITE_DB_PATH)
        return conn
    
    def connect(self):
        if self.DATABASE_TYPE == 'snowflake':
            conn = self.connect_to_snowflake()
        elif self.DATABASE_TYPE == 'sqlite':
            conn = self.connect_to_sqlite()
        else:
            raise Exception('Invalid database type, config.DATABASE_TYPE must be "snowflake" or "sqlite')
        return conn

    def __call__(self, sql: str):
        '''
            Pass a SQL query to the database and execute
            Return the output of the query if successful, otherwise 
            return the error message
        '''
        conn = self.connect()
        cs = conn.cursor()

        try:
            cs.execute(f"{sql}")
            columns = cs.fetchall()
            # Limit to 100 results
            if columns:
                columns = columns[:100]
            conn.close()
            return columns
        except Exception as e:
            conn.close()
            raise e
        
    def get_schema(self, database_name: str, schema: str=None, verbose: bool = False):
        conn = self.connect()
        cur = conn.cursor()

        # Execute a query to get the schema information
        if self.DATABASE_TYPE == 'snowflake':
            # Execute a query to get the schema information
            sql = f"SHOW TABLES IN DATABASE {database_name}"
            cur.execute(sql)
            columns = cur.fetchall()

            # Fetch the schema for every Table in database_name 
            schema_dict = []
            for column in columns:
                if column[3] == schema:
                    # List all Table names in the database
                    table_name = column[1]
                    # print(table_name)

                    tbl_dict = {}
                    tbl_dict["table"] = f"{schema}.{table_name}"
                    if verbose: print(f"Schema for {table_name}:")
                    
                    cur.execute(f"SHOW COLUMNS IN {schema}.{table_name}")
                    for row in cur.fetchall():
                        column_name = row[2]
                        data_type = row[3]
                        if verbose: print(f"{column_name}: {data_type}")
                        tbl_dict[column_name] = data_type
                    if verbose: print("\n")
                    schema_dict.append(tbl_dict)       
        
        elif self.DATABASE_TYPE == 'sqlite':
            sql = "SELECT name FROM sqlite_master WHERE type='table'"
            cur.execute(sql)
            tables = cur.fetchall()

            schema_dict = []
            for table_name in tables:
                table_name = table_name[0]
                
                tbl_dict = {}
                tbl_dict["table"] = f"{table_name}"
                if verbose: print(f"Schema for {table_name}:")

                # Get the schema for every Table in database_name
                sql = f"PRAGMA table_info({table_name})"
                cur.execute(sql)
                columns = cur.fetchall()

                for column in columns:
                    column_name = column[1]
                    if verbose: print(f"{column_name}: {column[2:]}")
                    tbl_dict[column_name] = column[2:]
                if verbose: print("\n")
                schema_dict.append(tbl_dict)   

        # Close connection
        cur.close()
        conn.close()

        # Convert json to string
        schema_str = json.dumps(schema_dict)

        # Save Database Schema to file
        fn = f'{database_name}_schema.json'
        with open(f'{fn}', 'w') as file:
            json.dump(schema_dict, file)
        print(f'Schema saved to {fn}')
        return schema_str
