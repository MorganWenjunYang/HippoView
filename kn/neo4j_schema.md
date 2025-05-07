# neo4j_schema.py (adapt on Ohio paper https://www.nature.com/articles/s41598-022-08454-z)

# 1. define the schema for the neo4j database
# 2. create the schema in the neo4j database
# 3. load the data into the neo4j database


## schema
### Nodes
    * Study
        - nct_id
        - Study Type
        - overall status
        - phase
        - start_date
        - completion_date
        - enrollment
        - brief_summary
        - allocation
        - masking
    * condition
        - condition_name
    * intervention
        - intervention_name
    * outcome
        - outcome_type
        - title
        - description
        - time_frame
        - population
        - units

    * sponsor
        - agency_class
        - lead_or_collaborator
        - name


### Edges
    * condition_study
    * intervention_study
    * study_outcome
    * sponsor_study
    * condition_intervention
    * condition_outcome
    * intervention_outcome

