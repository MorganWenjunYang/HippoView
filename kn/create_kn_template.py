# copy from https://github.com/biocypher/igan/blob/main/create_knowledge_graph.py
from biocypher import BioCypher
from igan.adapters.clinicaltrials_adapter import (
    ClinicalTrialsAdapter,
)

bc = BioCypher()
adapter = ClinicalTrialsAdapter()

bc.write_nodes(adapter.get_nodes())
bc.write_edges(adapter.get_edges())

bc.write_import_call()
bc.summary()