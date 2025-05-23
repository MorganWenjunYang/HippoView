#!/bin/bash

# This script runs in the Neo4j container (no Python available)
echo "Starting HippoView Knowledge Graph container"

# Start with pre-built knowledge graph
echo "Using pre-built knowledge graph"
echo "Copying biocypher output to Neo4j import directory"

# Make sure the import directory exists
mkdir -p /var/lib/neo4j/import/

# Copy files with error handling
cp -r /usr/app/biocypher-out/* /var/lib/neo4j/import/

# Apply Neo4j entrypoint patch
if [ -f /usr/app/biocypher_entrypoint_patch.sh ]; then
    echo "Applying BioCypher entrypoint patch"
    cat /usr/app/biocypher_entrypoint_patch.sh | cat - /startup/docker-entrypoint.sh > /tmp/temp_entrypoint.sh
    mv /tmp/temp_entrypoint.sh /startup/docker-entrypoint.sh
    chmod +x /startup/docker-entrypoint.sh
else
    echo "Warning: biocypher_entrypoint_patch.sh not found, continuing without patching"
fi

# Start Neo4j
echo "Starting Neo4j"
exec /startup/docker-entrypoint.sh neo4j 