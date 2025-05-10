# adapt from https://github.com/biocypher/igan/blob/main/create_knowledge_graph.py
from biocypher import BioCypher
from bc_adaptor_mongodb import (
    ClinicalTrialsMGDBAdapter,
)

# Initialize BioCypher with schema configuration
bc = BioCypher(
    schema_config_path="kn/kn_schema.yaml",
    output_directory="biocypher-out"
)

# Initialize adapter with all node and edge types
adapter = ClinicalTrialsMGDBAdapter()

# Write nodes and edges
bc.write_nodes(adapter.get_nodes())
bc.write_edges(adapter.get_edges())

# Generate import call and summary
bc.write_import_call()
bc.summary()