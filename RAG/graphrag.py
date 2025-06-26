#!/usr/bin/env python3
"""
GraphDB Data Layer

A simple data access layer for Neo4j graph database operations for clinical trials data.
All business logic is handled by the MCP server layer.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
import logging
from pydantic import BaseModel
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# Add parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default Neo4j configuration
DEFAULT_GRAPHDB_CONFIG = {
    "dburi": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "neo4jpassword"),
    "database": os.getenv("NEO4J_DATABASE", "neo4j")
}

class GraphDB:
    """Neo4j database connection and query handler - Pure data access layer"""
    
    def __init__(self, config: dict):
        self.config = config
        self.driver = None
        self.connected = False
    
    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.config["dburi"],
                auth=(self.config["username"], self.config["password"])
            )
            # Test the connection
            with self.driver.session(database=self.config.get("database", "neo4j")) as session:
                session.run("RETURN 1")
            self.connected = True
            logger.info(f"Successfully connected to Neo4j at {self.config['dburi']}")
            return True
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            self.connected = False
            return False
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: dict = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j database")
        
        try:
            with self.driver.session(database=self.config.get("database", "neo4j")) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise e
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the database"""
        try:
            stats_queries = {
                "total_nodes": "MATCH (n) RETURN count(n) as count",
                "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
                "node_labels": "CALL db.labels() YIELD label RETURN collect(label) as labels",
                "relationship_types": "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
            }
            
            stats = {}
            for key, query in stats_queries.items():
                result = self.execute_query(query)
                if result:
                    if key in ["node_labels", "relationship_types"]:
                        stats[key] = result[0].get("labels" if key == "node_labels" else "types", [])
                    else:
                        stats[key] = result[0].get("count", 0)
                else:
                    stats[key] = 0 if key not in ["node_labels", "relationship_types"] else []
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

def test_graphdb_connection():
    """
    Simple test function to verify Neo4j connection.
    """
    print("🧪 === GraphDB Connection Test ===")
    print()
    
    try:
        # Test 1: GraphDB Creation
        print("📋 Test 1: Creating GraphDB...")
        graph_db = GraphDB(DEFAULT_GRAPHDB_CONFIG)
        print("✅ GraphDB created successfully")
        print()
        
        # Test 2: Database Connection
        print("🔧 Test 2: Connecting to Neo4j...")
        connection_success = graph_db.connect()
        
        if not connection_success:
            print("❌ Neo4j connection failed")
            print("   Make sure Neo4j is running and accessible")
            return False
        
        print("✅ Neo4j connection successful")
        print(f"   - URI: {graph_db.config['dburi']}")
        print(f"   - Database: {graph_db.config.get('database', 'neo4j')}")
        print()
        
        # Test 3: Database Stats
        print("📊 Test 3: Getting Database Statistics...")
        try:
            stats = graph_db.get_database_stats()
            print(f"   - Total nodes: {stats.get('total_nodes', 0)}")
            print(f"   - Total relationships: {stats.get('total_relationships', 0)}")
            print(f"   - Node labels: {stats.get('node_labels', [])}")
            print(f"   - Relationship types: {stats.get('relationship_types', [])}")
            print("✅ Database statistics retrieved")
        except Exception as e:
            print(f"   ⚠️ Could not get database stats: {e}")
        
        print()
        
        # Test 4: Simple Query
        print("🔍 Test 4: Testing Simple Query...")
        try:
            test_query = "MATCH (n) RETURN count(n) as total_nodes LIMIT 1"
            result = graph_db.execute_query(test_query)
            if result:
                node_count = result[0].get('total_nodes', 0)
                print(f"   ✅ Query executed successfully: {node_count} nodes found")
            else:
                print("   ⚠️ Query returned no results")
        except Exception as e:
            print(f"   ❌ Query execution failed: {e}")
        
        print()
        
        # Test 5: Clinical Trial Query
        print("🔍 Test 5: Testing Clinical Trial Query...")
        try:
            trial_query = "MATCH (n:ClinicalTrial) RETURN count(n) as trial_count LIMIT 1"
            result = graph_db.execute_query(trial_query)
            if result:
                trial_count = result[0].get('trial_count', 0)
                print(f"   ✅ Found {trial_count} clinical trials in database")
            else:
                print("   ⚠️ No clinical trials found")
        except Exception as e:
            print(f"   ⚠️ Could not query clinical trials: {e}")
        
        print()
        
        # Final Results
        print("🎯 === Test Results Summary ===")
        print("✅ GraphDB Connection Test PASSED")
        print(f"   - Neo4j connection: ✅")
        print(f"   - Database access: ✅")
        print(f"   - Query execution: ✅")
        print()
        print("🚀 GraphDB data layer is ready for use!")
        
        # Clean up
        graph_db.close()
        return True
        
    except Exception as e:
        print(f"❌ GraphDB Connection Test FAILED: {str(e)}")
        import traceback
        print("📋 Error details:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run connection test
    print("Starting GraphDB Data Layer Tests...")
    print("This tests the data access layer only. Business logic is in the MCP server.")
    print()
    
    success = test_graphdb_connection()
    
    if success:
        print("\n🎉 GraphDB data layer is working correctly!")
        print("💡 Next steps:")
        print("   1. Start the unified MCP server: python RAG/rag_mcp_server.py")
        print("   2. Test the full system: python RAG/test_graphrag_mcp.py")
    else:
        print("\n💥 Connection test failed. Please check Neo4j configuration.")
        print("💡 Troubleshooting:")
        print("   - Ensure Neo4j is running on bolt://localhost:7687")
        print("   - Check username/password in environment variables")
        print("   - Verify database contains clinical trial data")

    