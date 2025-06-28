#! /usr/bin/env python3
# adapt from https://github.com/biocypher/igan/blob/main/create_knowledge_graph.py

import argparse
import pandas as pd
from pathlib import Path
from biocypher import BioCypher
from bc_adaptor_ctgovAPI import ClinicalTrialsAdapter
from config.config_manager import load_trial_config
from zzzzz_bc_adaptor_mongodb import ClinicalTrialsMGDBAdapter
import concurrent.futures

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Create knowledge graph from clinical trials data and export filtered NCT IDs')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-source', type=str, choices=['api'], 
                       help='Data source to use (overrides config)')
    parser.add_argument('--condition', type=str, help='Filter by condition')
    parser.add_argument('--study-type', type=str, help='Filter by study type (e.g., INTERVENTIONAL, OBSERVATIONAL)')
    parser.add_argument('--max-trials', type=int, help='Maximum number of trials to process')
    parser.add_argument('--export-nct-ids', type=str, default='data/filtered_nct_ids.csv',
                       help='Export filtered NCT IDs to CSV file (default: data/filtered_nct_ids.csv)')
    parser.add_argument('--no-export-nct-ids', action='store_true',
                       help='Skip exporting NCT IDs to CSV file')
    args = parser.parse_args()

    # Load configuration
    config = load_trial_config(args.config)
    
    # Override config with command line arguments
    config_updates = {}
    if args.data_source:
        config_updates['data_source'] = {'primary': args.data_source}
    if args.condition or args.study_type:
        api_filters = config_updates.get('api_filters', {})
        if args.condition:
            api_filters['condition'] = args.condition
        if args.study_type:
            api_filters.setdefault('advanced_filters', {})['study_type'] = args.study_type
        config_updates['api_filters'] = api_filters
    if args.max_trials:
        config_updates['sampling'] = {'max_trials': args.max_trials}
    
    if config_updates:
        config.update_config(config_updates)

    print(f"Creating knowledge graph with configuration:")
    print(f"  - Data source: {config.data_source.primary}")
    print(f"  - Condition filter: {config.api_filters.condition}")
    print(f"  - Study type: {config.api_filters.advanced_filters.get('study_type', 'all')}")
    print(f"  - Max trials: {config.sampling.max_trials}")
    print(f"  - Node types: {config.get_knowledge_graph_config().get('include_nodes', 'all')}")

    # Initialize BioCypher with schema configuration
    bc = BioCypher(
        schema_config_path="kn/kn_schema.yaml",
        output_directory="biocypher-out",
        biocypher_config_path="kn/bc_config.yaml",
        strict_mode=False,
    )

    # bc.show_ontology_structure(full=True)

    # Initialize adapter based on configuration
    if config.data_source.primary == 'mongodb':
        print("Using MongoDB adapter...")
        adapter = ClinicalTrialsMGDBAdapter()
    else:
        print("Using ClinicalTrials.gov API adapter...")
        adapter = ClinicalTrialsAdapter(config_path=args.config)

    # Create interceptor to capture NCT IDs while BioCypher processes nodes
    nct_ids = set()
    node_count = 0
    edge_count = 0
    
    def node_interceptor(nodes_generator):
        """Generator that yields nodes while capturing NCT IDs"""
        nonlocal node_count
        for node in nodes_generator:
            node_count += 1
            
            # Extract NCT ID from study nodes
            if isinstance(node, tuple) and len(node) >= 3:
                node_id, node_type, node_props = node[0], node[1], node[2]
                if node_type == "study" and node_id:
                    nct_ids.add(node_id)
            
            if node_count % 1000 == 0:
                print(f"  Processed {node_count} nodes, found {len(nct_ids)} unique NCT IDs")
            
            yield node
    
    def edge_interceptor(edges_generator):
        """Generator that yields edges while counting them"""
        nonlocal edge_count
        for edge in edges_generator:
            edge_count += 1
            
            if edge_count % 1000 == 0:
                print(f"  Processed {edge_count} edges")
            
            yield edge

    print("Processing nodes and collecting NCT IDs...")
    # Write nodes using interceptor
    bc.write_nodes(node_interceptor(adapter.get_nodes()))
    
    print(f"Processing edges...")
    # Write edges using interceptor
    bc.write_edges(edge_interceptor(adapter.get_edges()))

    # Export NCT IDs to CSV if requested
    if not args.no_export_nct_ids and nct_ids:
        nct_ids_list = sorted(list(nct_ids))
        output_path = Path(args.export_nct_ids)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({'nct_id': nct_ids_list})
        df.to_csv(args.export_nct_ids, index=False)
        
        print(f"\n📋 Exported {len(nct_ids_list)} NCT IDs to: {args.export_nct_ids}")
        print(f"   Use this file with: python data/load2mongo.py --nct-ids-file {args.export_nct_ids}")

    # Generate import call and summary
    bc.write_import_call()
    bc.summary()
    
    print(f"\nKnowledge graph creation completed!")
    print(f"\nProcessed {node_count} nodes and {edge_count} edges")
    if not args.no_export_nct_ids and nct_ids:
        print(f"\nExported {len(nct_ids)} NCT IDs to {args.export_nct_ids}")
    print(f"Import script generated: ./biocypher-out/neo4j-admin-import-call.sh")

if __name__ == "__main__":
    main()
