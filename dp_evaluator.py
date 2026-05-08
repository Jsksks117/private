import pandas as pd
import numpy as np
import duckdb
from scipy.stats import ks_2samp, entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple

class DPSyntheticEvaluator:
    def __init__(self, real_tables: Dict[str, pd.DataFrame], synth_tables: Dict[str, pd.DataFrame], seed: int = 42):
        """
        Comprehensive evaluation suite for Differentially Private Relational Data.
        
        Args:
            real_tables: Dictionary of original real dataframes.
            synth_tables: Dictionary of generated synthetic dataframes.
        """
        self.real = real_tables
        self.synth = synth_tables
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Verify schema match conceptually
        for table in self.real.keys():
            if table not in self.synth:
                raise ValueError(f"Missing synthetic table: {table}")

    # ==========================================
    # 1. Per-Table Fidelity (Marginal Stats)
    # ==========================================
    def evaluate_marginal_fidelity(self) -> Dict[str, Dict[str, float]]:
        """
        Computes KS statistic (continuous) and Jensen-Shannon Divergence (categorical).
        Lower is better (0.0 = perfect match).
        """
        results = {}
        for table, df_real in self.real.items():
            df_synth = self.synth[table]
            table_metrics = {}
            
            for col in df_real.columns:
                if col == 'id' or col.endswith('_id'): # Skip primary/foreign keys
                    continue
                    
                # Determine type heuristically (in practice, pass a schema)
                if pd.api.types.is_numeric_dtype(df_real[col]) and df_real[col].nunique() > 20:
                    # Continuous -> Kolmogorov-Smirnov Test
                    stat, _ = ks_2samp(df_real[col].dropna(), df_synth[col].dropna())
                    table_metrics[f"KS_{col}"] = stat
                else:
                    # Categorical -> Jensen-Shannon Divergence
                    # Get probability distributions
                    p_real = df_real[col].value_counts(normalize=True)
                    p_synth = df_synth[col].value_counts(normalize=True)
                    
                    # Align indices to ensure same categorical bins exist
                    all_categories = list(set(p_real.index).union(set(p_synth.index)))
                    p_real = p_real.reindex(all_categories, fill_value=0.0)
                    p_synth = p_synth.reindex(all_categories, fill_value=0.0)
                    
                    # JSD = H(M) - 0.5*H(P) - 0.5*H(Q) where M = 0.5*(P+Q)
                    m = 0.5 * (p_real + p_synth)
                    jsd = entropy(m) - 0.5 * entropy(p_real) - 0.5 * entropy(p_synth)
                    # Handle minor floating point negatives
                    table_metrics[f"JSD_{col}"] = max(0.0, jsd) 
                    
            results[table] = table_metrics
            
        return results

    # ==========================================
    # 2. Cross-Table Fidelity (Relational Integrity via DuckDB)
    # ==========================================
    def evaluate_relational_integrity(self, relationships: List[Tuple[str, str, str, str]]) -> Dict[str, float]:
        """
        Utilizes DuckDB's in-memory analytical engine to compute Join cardinality 
        and check foreign key violation rates.
        
        Args:
            relationships: List of tuples (parent_table, parent_key, child_table, child_key)
        """
        results = {}
        con = duckdb.connect(database=':memory:')

        # Explicitly register DataFrames so queries are stable across runtimes.
        for table_name, df in self.synth.items():
            con.register(f"synth_{table_name}", df)

        for table_name, df in self.real.items():
            con.register(f"real_{table_name}", df)

        for parent, p_key, child, c_key in relationships:
            rel_name = f"{parent}->{child}"
            
            # --- 2A. FK Violation Rate (Orphans) ---
            query = f"""
                SELECT COUNT(*) as orphans
                FROM synth_{child} c
                LEFT JOIN synth_{parent} p ON c.{c_key} = p.{p_key}
                WHERE p.{p_key} IS NULL
            """
            orphans = con.execute(query).df()['orphans'][0]
            total_children = len(self.synth[child])
            results[f"{rel_name}_FK_violation_rate"] = orphans / total_children if total_children > 0 else 0.0

            # --- 2B. Cardinality Ratio Preservation ---
            # How many children does each parent have on average?
            query_synth = f"""
                SELECT AVG(child_count) as avg_cardinality FROM (
                    SELECT p.{p_key}, COUNT(c.{c_key}) as child_count
                    FROM synth_{parent} p
                    LEFT JOIN synth_{child} c ON p.{p_key} = c.{c_key}
                    GROUP BY p.{p_key}
                )
            """
            
            query_real = f"""
                SELECT AVG(child_count) as avg_cardinality FROM (
                    SELECT p.{p_key}, COUNT(c.{c_key}) as child_count
                    FROM real_{parent} p
                    LEFT JOIN real_{child} c ON p.{p_key} = c.{c_key}
                    GROUP BY p.{p_key}
                )
            """
            
            synth_card = con.execute(query_synth).df()['avg_cardinality'][0]
            real_card = con.execute(query_real).df()['avg_cardinality'][0]
            
            synth_card = 0 if pd.isna(synth_card) else synth_card
            real_card = 0 if pd.isna(real_card) else real_card
            
            # Ratio distance from 1.0 (perfect preservation)
            results[f"{rel_name}_cardinality_ratio"] = synth_card / real_card if real_card > 0 else 0.0
            
        con.close()
        return results

    # ==========================================
    # 3. Downstream ML Utility (TSTR)
    # ==========================================
    def evaluate_tstr(self, target_table: str, target_col: str) -> Dict[str, float]:
        """
        Train on Synthetic, Test on Real (TSTR) protocol for a binary classification task.
        """
        df_real = self.real[target_table].copy()
        df_synth = self.synth[target_table].copy()
        
        # Ensure target column exists
        if target_col not in df_real.columns:
            raise ValueError(f"Target column '{target_col}' not found in '{target_table}'.")

        drop_id_cols = [col for col in df_real.columns if col == 'id' or col.endswith('_id')]
        if target_col in drop_id_cols:
            raise ValueError(f"Target column '{target_col}' cannot be an id/fk column.")

        y_real_raw = df_real[target_col]
        y_synth_raw = df_synth[target_col]

        X_real_raw = df_real.drop(columns=drop_id_cols + [target_col], errors='ignore')
        X_synth_raw = df_synth.drop(columns=drop_id_cols + [target_col], errors='ignore')

        X_real = pd.get_dummies(X_real_raw).fillna(0)
        X_synth = pd.get_dummies(X_synth_raw).fillna(0)

        feature_cols = sorted(set(X_real.columns).intersection(set(X_synth.columns)))
        if not feature_cols:
            raise ValueError("No overlapping encoded feature columns between real and synthetic tables.")
        X_real = X_real[feature_cols]
        X_synth = X_synth[feature_cols]

        # Build a shared label mapping across real and synthetic targets.
        all_labels = pd.Index(pd.concat([y_real_raw, y_synth_raw], axis=0).dropna().astype(str).unique())
        if len(all_labels) < 2:
            raise ValueError("Target has fewer than 2 classes after preprocessing.")
        label_to_id = {label: idx for idx, label in enumerate(sorted(all_labels.tolist()))}

        y_real = y_real_raw.astype(str).map(label_to_id)
        y_synth = y_synth_raw.astype(str).map(label_to_id)

        y_synth_unique = y_synth.dropna().nunique()
        if y_synth_unique < 2:
            raise ValueError("Synthetic target collapsed to one class; TSTR classification is not informative.")

        stratify_y = y_real if y_real.nunique() > 1 else None
        _, X_test, _, y_test = train_test_split(
            X_real,
            y_real,
            test_size=0.3,
            random_state=42,
            stratify=stratify_y,
        )
        
        # Train ML Model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_synth, y_synth)
        
        # Evaluate
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        
        # Handle binary or multiclass AUC
        if len(np.unique(y_test)) == 2:
            if probs.shape[1] < 2:
                raise ValueError("Classifier predicted a single probability column; cannot compute binary AUC.")
            auc = roc_auc_score(y_test, probs[:, 1])
        else:
            auc = roc_auc_score(y_test, probs, multi_class='ovr')
            
        return {
            "TSTR_Accuracy": accuracy_score(y_test, preds),
            "TSTR_AUC": auc
        }

    # ==========================================
    # 4. Empirical Privacy Audit (Shadow MIA & DCR)
    # ==========================================
    def evaluate_privacy(self, target_table: str) -> Dict[str, float]:
        """
        Measures privacy leakage empirically via Distance to Closest Record (DCR)
        and a baseline Membership Inference Attack.
        """
        df_real = self.real[target_table].copy()
        df_synth = self.synth[target_table].copy()
        
        # Preprocessing for distance calculation
        drop_cols = [col for col in df_real.columns if col == 'id' or col.endswith('_id')]
        df_real = pd.get_dummies(df_real.drop(columns=drop_cols)).fillna(0)
        df_synth = pd.get_dummies(df_synth.drop(columns=drop_cols)).fillna(0)
        
        # Align features
        common_cols = list(set(df_real.columns).intersection(set(df_synth.columns)))
        df_real = df_real[common_cols].values
        df_synth = df_synth[common_cols].values
        
        # --- 4A. Distance to Closest Record (DCR) ---
        # How close is the closest synthetic record to a real record?
        # If DCR is 0.0, a real record was explicitly memorized and copied.
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(df_synth)
        distances, _ = nn.kneighbors(df_real)
        mean_dcr = np.mean(distances)
        min_dcr = np.min(distances)
        
        # --- 4B. Baseline Membership Inference Attack (Shadow Model) ---
        # Create a holdout set (records NOT in the training data)
        # For simulation, we assume df_real represents the train set.
        # We generate a shadow holdout set of same size simply by adding noise.
        # In a real pipeline, df_real passed here should strictly be the train partition, 
        # and you would pass a separate true holdout set.
        holdout = df_real + self.rng.normal(0, 1.0, df_real.shape)
        
        # Attack setup: Predict 1 if the query record is closer to the Synthetic data
        # than a random guessing baseline threshold.
        # Distance feature for known members
        member_dists, _ = nn.kneighbors(df_real)
        # Distance feature for unknown non-members (holdout)
        non_member_dists, _ = nn.kneighbors(holdout)
        
        # Attacker data
        X_attack = np.vstack([member_dists, non_member_dists])
        # Labels: 1 = Member (Real), 0 = Non-Member (Holdout)
        y_attack = np.concatenate([np.ones(len(member_dists)), np.zeros(len(non_member_dists))])
        
        # Simple attacker model (thresholding distance)
        attack_clf = RandomForestClassifier(max_depth=2, random_state=42)
        attack_clf.fit(X_attack, y_attack)
        
        # Evaluate Attack Success (AUC)
        # If AUC is ~0.50, the attack fails (Perfect Privacy). 
        # If AUC > 0.60, the model leaks membership.
        attack_probs = attack_clf.predict_proba(X_attack)[:, 1]
        mia_auc = roc_auc_score(y_attack, attack_probs)

        return {
            "Mean_DCR_Euclidean": float(mean_dcr),
            "Min_DCR_Euclidean": float(min_dcr), # 0 indicates exact memorization violation
            "Membership_Inference_Attack_AUC": float(mia_auc)
        }


# ==========================================
# Example Usage Driver
# ==========================================
if __name__ == "__main__":
    print("Initializing Empirical Evaluation Suite...")
    
    # 1. Generate Mock Data (Simulating the output of the MultiRelationalDPGenerator)
    np.random.seed(42)
    
    # Real Flights
    real_flights = pd.DataFrame({
        'id': range(1, 101),
        'distance': np.random.normal(500, 150, 100),
        'airline_code': np.random.choice([0, 1, 2], 100),
        'is_delayed': np.random.choice([0, 1], 100)
    })
    
    # Synthetic Flights (With some DP noise added intentionally to simulate CTGAN output)
    synth_flights = pd.DataFrame({
        'id': range(1, 101),
        'distance': np.random.normal(520, 160, 100), 
        'airline_code': np.random.choice([0, 1, 2], 100, p=[0.4, 0.3, 0.3]),
        'is_delayed': np.random.choice([0, 1], 100, p=[0.6, 0.4])
    })
    
    # Real Passengers
    real_passengers = pd.DataFrame({
        'id': range(1, 301),
        'flight_id': np.random.choice(range(1, 101), 300), # FK to Flights
        'ticket_price': np.random.uniform(100, 5000, 300)
    })
    
    # Synthetic Passengers (Inject some broken FKs to test Relational Integrity module)
    synth_passengers = pd.DataFrame({
        'id': range(1, 301),
        'flight_id': np.random.choice(range(50, 150), 300), # FKs 101-149 will be orphans
        'ticket_price': np.random.uniform(90, 5500, 300)
    })
    
    real_db = {'Flights': real_flights, 'Passengers': real_passengers}
    synth_db = {'Flights': synth_flights, 'Passengers': synth_passengers}
    
    # --- Execute Evaluation ---
    evaluator = DPSyntheticEvaluator(real_tables=real_db, synth_tables=synth_db)
    
    print("\n1. Marginal Fidelity (Per-Table):")
    mf = evaluator.evaluate_marginal_fidelity()
    for table, metrics in mf.items():
        print(f"  [{table}]")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
            
    print("\n2. Relational Integrity (Cross-Table via DuckDB):")
    # Define Foreign Key: Flights(id) <- Passengers(flight_id)
    ri = evaluator.evaluate_relational_integrity([('Flights', 'id', 'Passengers', 'flight_id')])
    for k, v in ri.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n3. Downstream ML Utility (TSTR):")
    tstr = evaluator.evaluate_tstr(target_table='Flights', target_col='is_delayed')
    for k, v in tstr.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n4. Empirical Privacy Audit:")
    pa = evaluator.evaluate_privacy(target_table='Flights')
    for k, v in pa.items():
        print(f"  {k}: {v:.4f}")
