# adapt data to trial2vec format
from trial2vec import Trial2Vec, load_demo_data
import pandas as pd
from langchain_core.embeddings import Embeddings
import numpy as np
from typing import Union, List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# trial2vec test data schema

# test_data = load_demo_data()
# print(test_data.keys())
# dict_keys(['x', 'fields', 'ctx_fields', 'tag', 'x_val', 'y_val'])

# print('x columns', test_data['x'].columns)
# x columns Index(['nct_id', 'description', 'title', 'intervention_name', 'disease',
#        'keyword', 'outcome_measure', 'criteria', 'reference',
#        'overall_status'],
#       dtype='object')

# print('x_val columns', test_data['x_val'].columns)
# x_val columns Index(['target_trial', 'rank_1', 'rank_2', 'rank_3', 'rank_4', 'rank_5',
#        'rank_6', 'rank_7', 'rank_8', 'rank_9', 'rank_10'],
#       dtype='object')

# print('y_val columns', test_data['y_val'].columns)
# y_val columns Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='object')

# print('fields', test_data['fields'])
# ['title', 'intervention_name', 'disease', 'keyword']

# print('ctx_fields', test_data['ctx_fields'])
# ctx_fields ['description', 'criteria']

# print('tag', test_data['tag'])
# tag 'nct_id

class Trial2VecEmbeddings(Embeddings):
    """Wrapper for Trial2Vec embeddings to make it compatible with LangChain."""
    
    _instance = None
    _model = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Trial2VecEmbeddings, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._model = None
    
    def _get_model(self):
        if self._model is None:
            from trial2vec import Trial2Vec
            self._model = Trial2Vec(device='cpu')
            self._model.from_pretrained()
        return self._model
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
        model = self._get_model()
        # Convert texts to trial2vec format
        embeddings = model.encode(texts)
        return list(embeddings.values())

    
    def embed_query(self, text):
        """Embed a query."""
        # return self.embed_documents([text])[0]
        model = self._get_model()
        return model.sentence_vector(text)[0]
    
class Trial2VecRetriever(BaseRetriever):
    """Custom retriever for Trial2Vec embeddings."""
    
    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
    
    def get_relevant_documents(self, query: Union[str, List[float]]):
        """Get documents relevant to the query."""
        if isinstance(query, str):
            # If query is a string, embed it first
            query_embedding = self.embeddings.embed_query(query)
        else:
            # If query is already an embedding, use it directly
            query_embedding = query
            
        # Use the encoded query to search the vector store
        return self.vectorstore.similarity_search_by_vector(query_embedding, k=5)
    
    def invoke(self, query: Union[str, List[float]]):
        """Invoke the retriever."""
        return self.get_relevant_documents(query)


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
                    item[t2v_key] = ', '.join([o.get('title', '') for o in val if 'title' in o])
                else:
                    item[t2v_key] = None
            elif mongo_key in ['keyword', 'condition', 'intervention']:
                val = doc.get(mongo_key)
                if isinstance(val, list): 
                    item[t2v_key] = ', '.join([str(o) for o in val if o])
                else:
                    item[t2v_key] = str(val)
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
    model = Trial2Vec(device='cpu')
    model.from_pretrained()
    # if the trial in pre-encoded, just fetch
    
    return model.encode(documents).values()

