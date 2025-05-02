# load2mongo.py

import pandas as pd
from sqlalchemy import create_engine
from utils import connect_to_mongo, connect_to_aact
from pathlib import Path
import os
from datetime import datetime, date

def convert_dates(item):
    for key, value in item.items():
        if isinstance(value, date):
            item[key] = datetime.combine(value, datetime.min.time())
    return item

def load2mongo(nct_ids):
    # Connect to MongoDB
    client = connect_to_mongo()
    if not client:
        print("Failed to connect to MongoDB")
        return
    
    # Connect to AACT using SQLAlchemy
    # db_url = os.getenv('AACT_DB_URL', 'postgresql://aact:aact@aact-db.ctti-clinicaltrials.org:5432/aact')
    # engine = create_engine(db_url)
    engine = connect_to_aact()
    if not engine:
        print("Failed to connect to AACT database")
        client.close()
        return
    
    try:
 
        # Base query for studies table
        studies_query = """
        SELECT 
            s.nct_id,
            s.brief_title,
            s.study_type,
            s.overall_status,
            s.why_stopped,
            s.start_date,
            s.completion_date,
            s.phase,
            s.enrollment
        FROM studies s
        WHERE s.nct_id IN %(nct_ids)s
        """
        
        # Query for brief summaries
        summaries_query = """
        SELECT nct_id, description as brief_summary
        FROM brief_summaries
        WHERE nct_id IN %(nct_ids)s
        """
        
        # Query for designs
        designs_query = """
        SELECT 
            nct_id, 
            allocation, 
            intervention_model, 
            primary_purpose, 
            time_perspective,
            masking,
            observational_model
        FROM designs
        WHERE nct_id IN %(nct_ids)s
        """
        
        # Query for eligibilities
        eligibilities_query = """
        SELECT 
            nct_id, 
            criteria, 
            gender, 
            minimum_age, 
            maximum_age, 
            healthy_volunteers,
            sampling_method
        FROM eligibilities
        WHERE nct_id IN %(nct_ids)s
        """
        
        # Query for browse conditions
        conditions_query = """
        SELECT nct_id, mesh_term as condition
        FROM browse_conditions
        WHERE nct_id IN %(nct_ids)s
        """
        
        # Query for browse interventions
        interventions_query = """
        SELECT nct_id, mesh_term as intervention
        FROM browse_interventions
        WHERE nct_id IN %(nct_ids)s
        """
        
        # Query for keywords
        keywords_query = """
        SELECT nct_id, name as keyword
        FROM keywords
        WHERE nct_id IN %(nct_ids)s
        """
        
        # Query for outcomes
        outcomes_query = """
        SELECT 
            nct_id, 
            outcome_type, 
            title, 
            description, 
            time_frame,
            population,
            units
        FROM outcomes
        WHERE nct_id IN %(nct_ids)s
        """
        
        # Query for sponsors
        sponsors_query = """
        SELECT 
            nct_id, 
            agency_class, 
            lead_or_collaborator, 
            name
        FROM sponsors
        WHERE nct_id IN %(nct_ids)s
        """
        
        # Common parameters for all queries
        params = {'nct_ids': tuple(nct_ids)}
        
        # Execute all queries
        try:
            studies_df = pd.read_sql_query(studies_query, engine, params=params)
            summaries_df = pd.read_sql_query(summaries_query, engine, params=params)
            designs_df = pd.read_sql_query(designs_query, engine, params=params)
            eligibilities_df = pd.read_sql_query(eligibilities_query, engine, params=params)
            conditions_df = pd.read_sql_query(conditions_query, engine, params=params)
            interventions_df = pd.read_sql_query(interventions_query, engine, params=params)
            keywords_df = pd.read_sql_query(keywords_query, engine, params=params)
            outcomes_df = pd.read_sql_query(outcomes_query, engine, params=params)
            sponsors_df = pd.read_sql_query(sponsors_query, engine, params=params)
        except Exception as e:
            print(f"Error executing SQL queries: {str(e)}")
            return
        
        # Initialize merged_df with studies_df
        if studies_df.empty:
            print("No studies found in AACT for the given NCT IDs")
            return
            
        merged_df = studies_df.copy()
        
        # Merge summaries if available
        if not summaries_df.empty:
            merged_df = merged_df.merge(summaries_df, on='nct_id', how='left')
        
        # Merge designs if available
        if not designs_df.empty:
            merged_df = merged_df.merge(designs_df, on='nct_id', how='left')
        
        # Merge eligibilities if available
        if not eligibilities_df.empty:
            merged_df = merged_df.merge(eligibilities_df, on='nct_id', how='left')
        
        # Group and merge conditions if available
        if not conditions_df.empty:
            conditions_grouped = conditions_df.groupby('nct_id')['condition'].apply(list).reset_index()
            merged_df = merged_df.merge(conditions_grouped, on='nct_id', how='left')
        
        # Group and merge interventions if available
        if not interventions_df.empty:
            interventions_grouped = interventions_df.groupby('nct_id')['intervention'].apply(list).reset_index()
            merged_df = merged_df.merge(interventions_grouped, on='nct_id', how='left')
        
        # Group and merge keywords if available
        if not keywords_df.empty:
            keywords_grouped = keywords_df.groupby('nct_id')['keyword'].apply(list).reset_index()
            merged_df = merged_df.merge(keywords_grouped, on='nct_id', how='left')
        
        # Handle outcomes if available
        if not outcomes_df.empty:
            outcomes_list = []
            for nct_id, group in outcomes_df.groupby('nct_id'):
                outcomes_list.append({
                    'nct_id': nct_id,
                    'outcomes': group.drop('nct_id', axis=1).to_dict('records')
                })
            outcomes_df = pd.DataFrame(outcomes_list)
            merged_df = merged_df.merge(outcomes_df, on='nct_id', how='left')
        
        # Handle sponsors if available
        if not sponsors_df.empty:
            sponsors_list = []
            for nct_id, group in sponsors_df.groupby('nct_id'):
                sponsors_list.append({
                    'nct_id': nct_id,
                    'sponsors': group.drop('nct_id', axis=1).to_dict('records')
                })
            sponsors_df = pd.DataFrame(sponsors_list)
            merged_df = merged_df.merge(sponsors_df, on='nct_id', how='left')
        
        # Convert DataFrame to list of dictionaries
        trials_data = merged_df.to_dict('records')
        
        # Insert into MongoDB
        db = client['clinical_trials']
        collection = db['trialgpt_trials']
        
        # Clear existing data for these NCT IDs
        collection.delete_many({"nct_id": {"$in": nct_ids}})
        
        # Insert new data
        if trials_data:
            # Convert all date objects to datetime
            trials_data = [convert_dates(item) for item in trials_data]
            collection.insert_many(trials_data)
            print(f"Successfully loaded {len(trials_data)} trials to MongoDB")
        else:
            print("No trials found in AACT for the given NCT IDs")
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
    finally:
        # Close connections
        # engine.dispose()
        engine.close()
        client.close()

if __name__ == "__main__":
    # Read NCT IDs from trialgpt.studies_list.csv
    studies_df = pd.read_csv(Path("./data/raw/trialgpt/trialgpt.studies_list.csv"))  # Get top 5 NCT IDs
       
    nct_ids = studies_df['nct_id'].head(1000).tolist()
    print(f"fetching {len(nct_ids)} trials: {nct_ids}")
    load2mongo(nct_ids)
