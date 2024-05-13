import psycopg2
import sqlite3

hostname='localhost'
database='postgres'
username='postgres'
pwd='harsha$23'
port_id=5434

source_conn = sqlite3.connect('C:/Users/harsh/Downloads/geodblp/geodblp.db')
source_cursor = source_conn.cursor()

source_table_name = 'pub_data_authoraffiliation'

source_cursor.execute(f"PRAGMA table_info({source_table_name});")
source_schema = source_cursor.fetchall()

source_cursor.execute(f"SELECT * FROM {source_table_name};")
data = source_cursor.fetchall()

source_conn.close()

dest_conn = psycopg2.connect( host=hostname,dbname=database,user=username,password=pwd,port=port_id)
dest_cursor = dest_conn.cursor()

create_table_query = f"CREATE TABLE if not exists {source_table_name} ("
for column in source_schema:
    name = column[1]
    data_type = column[2]
    not_null = "NOT NULL" if column[3] else ""
    default_value = f"DEFAULT {column[4]}" if column[4] is not None else ""
    constraints = ", ".join(filter(None, [not_null, default_value]))
    create_table_query += f"{name} {data_type} {constraints}, "
create_table_query = create_table_query.rstrip(", ") + ");"

dest_cursor.execute(create_table_query)
insert_query = f"INSERT INTO {source_table_name} VALUES ({', '.join(['%s' for _ in range(len(source_schema))])})"
dest_cursor.executemany(insert_query, data)

dest_conn.commit()
dest_conn.close()

print(f"Table '{source_table_name}' created and data inserted successfully in the destination database.")
