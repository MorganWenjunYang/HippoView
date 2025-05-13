import re
import psycopg2 
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from bson.json_util import dumps


load_dotenv(".env")
AACT_USER = os.getenv("AACT_USER")
AACT_PWD = os.getenv("AACT_PWD")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PWD = os.getenv("MONGO_PWD")
MONGO_URI = os.getenv("MONGO_URI")

def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


# AACT DB (postgres)

def connect_to_aact():
    conn = psycopg2.connect(
        database="aact",
        host="aact-db.ctti-clinicaltrials.org",
        user=AACT_USER,
        password=AACT_PWD,
        port=5432,
    )
    return conn


def aact_run_sql_query(conn, sql_query: str) -> list:
    cur = conn.cursor()
    cur.execute(sql_query)
    rows = cur.fetchall()
    cur.close()
    return rows


def get_list_tables(conn):
    sql_query = "SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE table_schema = 'ctgov'"
    response = aact_run_sql_query(conn, sql_query)
    tables = [r[2] for r in response]
    return tables


def get_columns_table(conn, table_name) -> list:
    sql_query = f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = '{table_name}';
    """
    response = aact_run_sql_query(conn, sql_query)
    return response


def get_enumerated_col_values(conn, table, col):
    sql_query = f"""
    SELECT distinct {col} FROM {table}
    """
    response = aact_run_sql_query(conn, sql_query)
    response = set([x[0] for x in response]) - set([" ", "", None])
    response = list(response)
    return response


# MONGO DB

def connect_to_mongo():
    try:
        client = MongoClient(MONGO_URI, username=MONGO_USER, password=MONGO_PWD)
        print("Connected to MongoDB")
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None
    
def fetch_all_documents(collection):
    client = connect_to_mongo()
    db = client["clinical_trials"]
    collection = db["trialgpt_trials"]
    cursor = collection.find()
    json_docs = [dumps(doc) for doc in cursor]
    return json_docs

def fetch_document_by_nct_id(nct_id):
    client = connect_to_mongo()
    db = client["clinical_trials"]
    collection = db["trialgpt_trials"]
    cursor = collection.find_one({"nct_id": nct_id})
    json_doc = dumps(cursor, indent=2)
    return json_doc

def fetch_documents_by_sponsor_name(sponsor_name):
    client = connect_to_mongo()
    db = client["clinical_trials"]
    collection = db["trialgpt_trials"]
    cursor = collection.find({"sponsors.name": sponsor_name})
    json_docs = [dumps(doc, indent=2) for doc in cursor]
    return json_docs

# NEO4J DB

# def connect_to_neo4j():
#     driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
#     return driver

