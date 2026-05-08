import networkx as nx
import math
from typing import Dict, List, Any

class SchemaBudgetAllocator:
    def __init__(self, schema: Dict[str, List[str]], total_epsilon: float):
        """
        Initializes the budget allocator.
        
        Args:
            schema: A dictionary representing the database schema where:
                    - Keys are table names
                    - Values are lists of parent tables (foreign key references)
            total_epsilon: The overall differential privacy budget (epsilon) to allocate.
        """
        self.schema = schema
        self.total_epsilon = total_epsilon
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Constructs a Directed Acyclic Graph (DAG) from the schema."""
        for table, parents in self.schema.items():
            self.graph.add_node(table)
            for parent in parents:
                # Edge goes from Parent -> Child
                # This naturally captures the top-down dependency of data generation
                self.graph.add_edge(parent, table)
        
        # Verify the dependency graph is a DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("The provided database schema contains cyclic dependencies and is not a valid DAG.")

    def calculate_downstream_fanout(self) -> Dict[str, int]:
        """Calculates the downstream fan-out (total count of transitively affected tables)."""
        fanout = {}
        for node in self.graph.nodes:
            # nx.descendants returns all nodes reachable from the current node
            descendants = nx.descendants(self.graph, node)
            fanout[node] = len(descendants)
        return fanout

    def allocate_budget(self) -> Dict[str, float]:
        """Calculates the allocated budget per table based on downstream fan-out."""
        fanout = self.calculate_downstream_fanout()
        weights = {}
        total_weight = 0.0

        # Calculate weight(T) = 1 + log(1 + downstream_fanout(T))
        for table, f_out in fanout.items():
            w = 1.0 + math.log(1.0 + f_out)
            weights[table] = w
            total_weight += w

        # Normalize weights and allocate epsilon budget proportionally
        allocation = {}
        for table, w in weights.items():
            normalized_weight = w / total_weight
            allocation[table] = normalized_weight * self.total_epsilon

        return allocation, fanout, weights

    def get_generation_order(self) -> List[str]:
        """Returns a topological sort of the tables to ensure parents are generated first."""
        return list(nx.topological_sort(self.graph))

    def run(self) -> Dict[str, Any]:
        """Executes the full pipeline and returns all metrics."""
        generation_order = self.get_generation_order()
        allocation, fanout, weights = self.allocate_budget()
        
        return {
            "generation_order": generation_order,
            "epsilon_allocation": allocation,
            "fanout_counts": fanout,
            "weights": weights
        }


if __name__ == "__main__":
    # Example Usage: Defining a multi-relational database schema
    # Parent tables are listed for each child table representing the foreign keys
    example_schema = {
        'Flights': [],
        'Airlines': [],
        'Passengers': ['Flights'],
        'Tickets': ['Passengers', 'Airlines'],
        'Reviews': ['Passengers', 'Airlines']
    }
    
    TOTAL_BUDGET = 10.0
    
    print(f"Allocating Total Epsilon: {TOTAL_BUDGET}")
    print("-" * 65)
    
    allocator = SchemaBudgetAllocator(example_schema, TOTAL_BUDGET)
    results = allocator.run()
    
    print("Topological Generation Order (Parents before Children):")
    for idx, table in enumerate(results['generation_order'], 1):
        print(f"  {idx}. {table}")
        
    print("\nDP Budget Allocation Overview:")
    print(f"{'Table':<15} | {'Fan-out':<8} | {'Weight':<10} | {'Epsilon Allocation':<15}")
    print("-" * 65)
    for table in results['generation_order']:
        f_out = results['fanout_counts'][table]
        w = results['weights'][table]
        eps = results['epsilon_allocation'][table]
        print(f"{table:<15} | {f_out:<8} | {w:<10.4f} | {eps:<15.4f}")
