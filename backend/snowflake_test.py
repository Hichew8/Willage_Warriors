import snowflake.connector

# Connect to Snowflake
conn = snowflake.connector.connect(
    user="Jquan115",
    password="Hacklytics2025",
    account="fyraoxz-flb98787"
)

# Create a cursor object
cursor = conn.cursor()

# Run SHOW DATABASES
#cursor.execute("SHOW DATABASES;")
cursor.execute("USE DATABASE OFFICE_VISITS_CPT_99213__99214_NEGOTIATED_RATES_NATIONAL_DATASETS;")
cursor.execute("USE SCHEMA SHARED;")
print("Checking available tables in schema 'SHARED'...")
cursor.execute("SHOW TABLES IN SHARED;")
print(cursor.fetchall())
# databases = cursor.fetchall()

# # Print databases
# for db in databases:
#     print(db)