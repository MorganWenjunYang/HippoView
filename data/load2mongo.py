# load2mongo.py

import pandas as pd
from utils import connect_to_aact, connect_to_mongo
from pathlib import Path

def load2mongo():
    # Connect to MongoDB
    client = connect_to_mongo()
    if not client:
        print("Failed to connect to MongoDB")
        return
    
    # Connect to AACT
    conn = connect_to_aact()
    if not conn:
        print("Failed to connect to AACT")
        return
    
    try:
        # Read NCT IDs from trialgpt.studies_list.csv
        studies_df = pd.read_csv(Path("./data/raw/trialgpt/trialgpt.studies_list.csv"))
        nct_ids = studies_df['nct_id'].head(5).tolist()  # Get top 5 NCT IDs
        
        # Query AACT for these trials
        query = """
        SELECT 
            s.nct_id,
            s.brief_title,
            s.official_title,
            s.overall_status,
            s.start_date,
            s.completion_date,
            s.phase,
            s.study_type,
            s.enrollment,
            s.condition,
            s.intervention,
            s.description,
            s.criteria
        FROM studies s
        WHERE s.nct_id = ANY(%s)
        """
        
        # Execute query
        trials_df = pd.read_sql_query(query, conn, params=(nct_ids,))
        
        # Convert DataFrame to list of dictionaries
        trials_data = trials_df.to_dict('records')
        
        # Insert into MongoDB
        db = client['clinical_trials']
        collection = db['trialgpt_trials']
        
        # Clear existing data for these NCT IDs
        collection.delete_many({"nct_id": {"$in": nct_ids}})
        
        # Insert new data
        if trials_data:
            collection.insert_many(trials_data)
            print(f"Successfully loaded {len(trials_data)} trials to MongoDB")
        else:
            print("No trials found in AACT for the given NCT IDs")
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
    finally:
        # Close connections
        conn.close()
        client.close()

if __name__ == "__main__":
    load2mongo()
