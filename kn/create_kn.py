# adapt from https://github.com/biocypher/igan/blob/main/create_knowledge_graph.py

from biocypher import BioCypher
from bc_adaptor_mongodb import ClinicalTrialsMGDBAdapter
from bc_adaptor_ctgovAPI import ClinicalTrialsAdapter
# from bc_adaptor import ClinicalTrialsAdapter
import concurrent.futures

# Initialize BioCypher with schema configuration

bc = BioCypher(
    schema_config_path="kn/kn_schema.yaml",
    output_directory="biocypher-out",
    biocypher_config_path="kn/bc_config.yaml",
    strict_mode=False,
)

# bc.show_ontology_structure(full=True)

# Initialize adapter with all node and edge types
# adapter = ClinicalTrialsMGDBAdapter()
adapter = ClinicalTrialsAdapter()   


# Write nodes and edges
bc.write_nodes(adapter.get_nodes())
bc.write_edges(adapter.get_edges())

# Generate import call and summary
bc.write_import_call()
bc.summary()
