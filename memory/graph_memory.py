from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password123")

driver = GraphDatabase.driver(URI, auth=AUTH)

def add_fact(entity1: str, relationship: str, entity2: str):
    """Add a relationship fact to the knowledge graph.
    Example: add_fact("User", "IS_BUILDING", "Multi-Agent System")
    """
    query = f"""
    MERGE (a:Entity {{name: $entity1}})
    MERGE (b:Entity {{name: $entity2}})
    MERGE (a)-[:{relationship}]->(b)
    """
    with driver.session() as session:
        session.run(query, entity1=entity1, entity2=entity2)

def get_facts_about(entity: str):
    """Retrieve all facts connected to a given entity."""
    query = """
    MATCH (a:Entity {name: $entity})-[r]->(b:Entity)
    RETURN a.name AS from_entity, type(r) AS relationship, b.name AS to_entity
    """
    with driver.session() as session:
        result = session.run(query, entity=entity)
        return [f"{record['from_entity']} {record['relationship']} {record['to_entity']}" for record in result]

if __name__ == "__main__":
    add_fact("User", "IS_BUILDING", "Multi_Agent_System")
    add_fact("Multi_Agent_System", "USES", "LangGraph")
    add_fact("Multi_Agent_System", "USES", "Qdrant")

    print("\nFacts about User:")
    for fact in get_facts_about("User"):
        print("-", fact)

    print("\nFacts about Multi_Agent_System:")
    for fact in get_facts_about("Multi_Agent_System"):
        print("-", fact)