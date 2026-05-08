import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List

# Import our previously built modules
from schema_budget_allocator import SchemaBudgetAllocator
from dp_ctgan import train_dp_ctgan, get_data_dim

import sqlalchemy
from sqlalchemy import create_engine

class MultiRelationalDPGenerator:
    def __init__(self, 
                 schema_def: Dict[str, List[str]], 
                 total_epsilon: float,
                 db_connection_string: str = "sqlite:///synthetic_db.sqlite"):
        """
        Coordinates the training and generation of a multi-relational synthetic database.
        
        Args:
            schema_def: Dictionary defining parent dependencies {table_name: [parent1, parent2]}
            total_epsilon: Total privacy budget for the entire database release
            db_connection_string: SQLAlchemy URI for saving the generated output
        """
        self.schema_def = schema_def
        self.total_epsilon = total_epsilon
        self.db_engine = create_engine(db_connection_string)
        
        # 1. Initialize our previously built NetworkX Allocator
        self.allocator = SchemaBudgetAllocator(schema_def, total_epsilon)
        alloc_results = self.allocator.run()
        
        self.generation_order = alloc_results['generation_order']
        self.epsilon_allocation = alloc_results['epsilon_allocation']
        
        # Store synthetic data locally as DataFrames during the pipeline before saving to DB
        self.synthetic_tables: Dict[str, pd.DataFrame] = {}
        
    def _mock_table_schema(self, table_name: str) -> List[Dict]:
        """
        Mock helper: Returns a dummy CTGAN schema definition for an arbitrary table 
        just so the DP_CTGAN models have correct structural dimensions to train on.
        In a real scenario, this would parse your actual dataframe columns.
        """
        # For simplicity of the orchestrator, assume every table has 1 continuous & 1 discrete column natively
        schema = [
            {'type': 'continuous', 'modes': 3},       # Generic value column
            {'type': 'discrete',   'output_dim': 2}   # Generic categorical column
        ]
        
        # If this table has parents, we must also reserve "columns" in its feature space
        # to hold the embedded parent foreign keys, enabling conditional generation.
        # We treat foreign keys as continuous features (embeddings) or highly dimensional discrete.
        for _ in self.schema_def.get(table_name, []):
             schema.append({'type': 'continuous', 'modes': 1}) # Simplified FK placeholder
             
        return schema

    def _fetch_real_data_for_training(self, table_name: str, ctgan_schema: List[Dict]) -> torch.Tensor:
        """
        Mock helper: Returns fake "real" data tensor to simulate training.
        In a real scenario, this would query your real Pandas DataFrame, join it with 
        its parent tables, and preprocess it into the continuous/discrete GMM tensors 
        that CTGAN requires.
        """
        # Simulate 1000 real rows to train on
        data_dim = get_data_dim(ctgan_schema)
        return torch.randn(1000, data_dim)

    def _decode_synthetic_tensor(self, tensor_data: torch.Tensor, ctgan_schema: List[Dict]) -> pd.DataFrame:
        """
        Mock helper: Converts the output PyTorch tensor back into a Pandas DataFrame.
        """
        df = pd.DataFrame(tensor_data.numpy())
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        
        # Add a primary key to this newly generated synthetic table
        df.insert(0, 'id', range(1, len(df) + 1))
        return df

    def train_and_generate(self, num_synthetic_rows: int = 500):
        """
        Main orchestration loop traversing the DAG top-down.
        """
        print(f"Starting Multi-Relational DP Generation Pipeline")
        print(f"Targeting Epsilon: {self.total_epsilon:.2f}\n")
        
        # 2. Iterate in perfect topological order (Parents -> Children)
        for table in self.generation_order:
            budget = self.epsilon_allocation[table]
            parents = self.schema_def.get(table, [])
            
            print(f"--- Processing Table: {table} ---")
            print(f"Allocated Budget ε: {budget:.4f}")
            print(f"Parent Dependencies: {parents}")
            
            # 3. Handle Foreign Key Integrity & Conditioning
            # If parents exist, we need their already-synthesized Primary Keys 
            # to condition this child table's generation.
            parent_context = None
            if parents:
                print(f"Loading PKs from synthesized parents to condition {table} generation...")
                # In a full implementation, you'd concatenate the synthetic parent data 
                # (or just their IDs) to form the condition context matrix [num_synthetic_rows, parent_dim]
                # parent_context = pd.concat([self.synthetic_tables[p]['id'] for p in parents])
                parent_context = torch.randn(num_synthetic_rows, len(parents)) # Mock context tensor
            
            # Prepare schema and mock real data
            ctgan_schema = self._mock_table_schema(table)
            real_data_tensor = self._fetch_real_data_for_training(table, ctgan_schema)
            
            # --- Train Phase ---
            print(f"Training DP-CTGAN for {table}...")
            # Here we pass the allocated budget directly into the PRV Accountant stop condition
            generator = train_dp_ctgan(
                real_data=real_data_tensor,
                column_info=ctgan_schema,
                epochs=10,             # Keep low for script demonstration
                batch_size=200,
                target_epsilon=budget, # <--- 2. Inject specific DP weight here!
                noise_multiplier=1.5,
                device='cpu'
            )
            
            # --- Generation Phase ---
            print(f"Generating {num_synthetic_rows} synthetic rows for {table}...")
            generator.eval()
            with torch.no_grad():
                # If conditional context exists (parents), we would append it to `z` 
                # or pass it to a ConditionalGenerator variant.
                # For this standard CTGAN generator, we'll just sample random Z.
                z = torch.randn(num_synthetic_rows, 128) 
                synthetic_tensor = generator(z)
                
            # Convert back to DataFrame
            df_synth = self._decode_synthetic_tensor(synthetic_tensor, ctgan_schema)
            
            # Note: For strict relational integrity, after decoding, you explicitly 
            # sample/assign Foreign Keys from `self.synthetic_tables[parent]['id']` 
            # into the child df_synth based on the conditional probabilities learned.
            
            self.synthetic_tables[table] = df_synth
            print(f"Finished {table}.\n")
            
        return self.synthetic_tables

    def save_to_database(self):
        """
        4. Use SQLAlchemy to persist the synthetic DataFrames into a relational database.
        """
        print(f"Saving synthetic tables to database: {self.db_engine.url}")
        
        # Save in the same topological order so Foreign Key constraints aren't violated upon insertion
        for table in self.generation_order:
            df = self.synthetic_tables[table]
            
            print(f"Writing {table} ({len(df)} rows)...")
            # Write to SQL. In a strict Postgres setup with complex FKs, you might need 
            # to temporarily disable triggers or ensure schemas are pre-created.
            df.to_sql(name=table.lower(), con=self.db_engine, if_exists='replace', index=False)
            
        print("Database save complete!")


if __name__ == "__main__":
    # Example Schema matching the NetworkX Allocator example
    schema = {
        'Flights': [],
        'Airlines': [],
        'Passengers': ['Flights'],
        'Tickets': ['Passengers', 'Airlines'],
        'Reviews': ['Passengers', 'Airlines']
    }
    
    # We will use an in-memory or file-based SQLite database for safety/demonstration
    db_uri = "sqlite:///multirelational_synthetic.db"
    
    pipeline = MultiRelationalDPGenerator(
        schema_def=schema,
        total_epsilon=10.0,
        db_connection_string=db_uri
    )
    
    # Run the full end-to-end pipeline
    synthetic_dfs = pipeline.train_and_generate(num_synthetic_rows=300)
    
    # Save the output to SQLAlchemy
    pipeline.save_to_database()
