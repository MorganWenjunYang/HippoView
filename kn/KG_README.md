# Knowledge Graph

This setup creates a Neo4j-based knowledge graph from clinical trials data.

## Local version

1. Create import file:

```bash
python3 ./kn/create_kn.py
```

2. Load the node/edge:
```bash
chmod +x ./biocypher-out/neo4j-admin-import-call.sh
./biocypher-out/neo4j-admin-import-call.sh
```

3. Check the WebUI:
```bash
http://localhost:7474
```


## Setup

1. Build the image:

```bash
./build.sh
```

2. In remote host:

```bash
docker run --pull always -d \
  --name biocypher-kg \
  --platform linux/amd64 \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4jpassword \
  -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
  {YOUR_IMAGE_REGISTRY}/biocypher-kg:{YOUR_TAG} 

```

## Access the Knowledge Graph

Once the container is running and the knowledge graph is built:

- **Neo4j Browser**: http://localhost:7474
- **Login credentials**: neo4j / password

## Example Queries

Try these Cypher queries in the Neo4j browser:
```

```

## Structure

The knowledge graph is built with these entities:

- **Studies**: Clinical trials
- **Conditions**: Medical conditions being studied
- **Interventions**: Treatments or procedures
- **Outcomes**: Measurements or results
- **Sponsors**: Organizations running the trials

And relationships between them:

- condition_has_study
- intervention_has_study
- study_has_outcome
- sponsor_has_study
- condition_has_intervention
- condition_has_outcome
- intervention_has_outcome

## Customization

To customize the knowledge graph schema, edit:
- `kn/kn_schema.yaml`
- `kn/bc_adaptor_xxx`

To customize the studies included or the scope, edit:
- `kn/bc_adaptor_ctgovAPI.py` -> QUERY_PARAMS

Then redo the above steps accordingly.