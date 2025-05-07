# adapt data to trial2vec format
from trial2vec import Trial2Vec, load_demo_data
import pandas as pd

# trial2vec test data schema

test_data = load_demo_data()
print(test_data.keys())
# dict_keys(['x', 'fields', 'ctx_fields', 'tag', 'x_val', 'y_val'])

print('x columns', test_data['x'].columns)
# x columns Index(['nct_id', 'description', 'title', 'intervention_name', 'disease',
#        'keyword', 'outcome_measure', 'criteria', 'reference',
#        'overall_status'],
#       dtype='object')

print('x_val columns', test_data['x_val'].columns)
# x_val columns Index(['target_trial', 'rank_1', 'rank_2', 'rank_3', 'rank_4', 'rank_5',
#        'rank_6', 'rank_7', 'rank_8', 'rank_9', 'rank_10'],
#       dtype='object')

print('y_val columns', test_data['y_val'].columns)
# y_val columns Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='object')

print('fields', test_data['fields'])
# ['title', 'intervention_name', 'disease', 'keyword']

print('ctx_fields', test_data['ctx_fields'])
# ctx_fields ['description', 'criteria']

print('tag', test_data['tag'])
# tag 'nct_id


def mongodb_data_adaptor(data: list) -> pd.DataFrame:
    '''
    Convert a list of MongoDB documents to a pandas DataFrame matching the trial2vec schema.

        mongodb document schema:
{
  "nct_id": "NCT01234567",
  "brief_title": "A Study of Something",
  "study_type": "Interventional",
  "overall_status": "Completed",
  "why_stopped": null,
  "start_date": "2020-01-01T00:00:00",
  "completion_date": "2021-01-01T00:00:00",
  "phase": "Phase 2",
  "enrollment": 100,
  "brief_summary": "This is a summary.",
  "allocation": "Randomized",
  "intervention_model": "Parallel Assignment",
  "primary_purpose": "Treatment",
  "time_perspective": null,
  "masking": "Double",
  "observational_model": null,
  "criteria": "Inclusion: ... Exclusion: ...",
  "gender": "All",
  "minimum_age": "18 Years",
  "maximum_age": "65 Years",
  "healthy_volunteers": "No",
  "sampling_method": "Probability Sample",
  "condition": ["Diabetes Mellitus", "Hypertension"],
  "intervention": ["Metformin", "Placebo"],
  "keyword": ["Blood Sugar", "Insulin"],
  "outcomes": [
    {
      "outcome_type": "Primary",
      "title": "Change in HbA1c",
      "description": "Measured at 6 months",
      "time_frame": "6 months",
      "population": "Adults",
      "units": "%"
    }
  ],
  "sponsors": [
    {
      "agency_class": "Industry",
      "lead_or_collaborator": "Lead",
      "name": "PharmaCorp"
    }
  ]
}

    trial2vec document schema:
    columns Index(['nct_id', 'description', 'title', 'intervention_name', 'disease',
        'keyword', 'outcome_measure', 'criteria', 'reference',
        'overall_status'
    '''
    # convert mongodb data to trial2vec encodable format

    # Define mapping from MongoDB schema to trial2vec schema
    mongo_to_t2v = {
        'nct_id': 'nct_id',
        'brief_summary': 'description',
        'brief_title': 'title',
        'intervention': 'intervention_name',
        'condition': 'disease',
        'keyword': 'keyword',
        'outcomes': 'outcome_measure',
        'criteria': 'criteria',
        'reference': 'reference',  # Not present in mongo, will be None
        'overall_status': 'overall_status',
    }
    # Prepare list of dicts with correct keys
    converted = []
    for doc in data:
        item = {}
        for mongo_key, t2v_key in mongo_to_t2v.items():
            if mongo_key == 'outcomes':
                # Convert list of dicts to a string or summary (or keep as is)
                val = doc.get('outcomes')
                if isinstance(val, list):
                    # Join outcome titles if present
                    item[t2v_key] = '; '.join([o.get('title', '') for o in val if 'title' in o])
                else:
                    item[t2v_key] = None
            elif mongo_key == 'reference':
                # Not present in mongo, set as None
                item[t2v_key] = None
            else:
                item[t2v_key] = doc.get(mongo_key)
        converted.append(item)
    # Create DataFrame
    df = pd.DataFrame(converted)
    # Ensure all columns are present
    t2v_columns = ['nct_id', 'description', 'title', 'intervention_name', 'disease',
                   'keyword', 'outcome_measure', 'criteria', 'reference', 'overall_status']
    for col in t2v_columns:
        if col not in df.columns:
            df[col] = None
    df = df[t2v_columns]
    return {'x': df}

def ctgov_data_adaptor(data):
    # convert data to trial2vec format
    return data

def t2v_embed_documents(documents):
    model = Trial2Vec()
    model.from_pretrained()

    return model.encode(documents)

