from __future__ import annotations

from typing import Dict, List, Tuple
import re

import networkx as nx
import pandas as pd


def _infer_primary_key(df: pd.DataFrame) -> str:
    if "id" in df.columns and df["id"].is_unique and df["id"].notna().all():
        return "id"

    for column in df.columns:
        series = df[column]
        if series.is_unique and series.notna().all():
            return column

    return df.columns[0]


def _overlap_ratio(child_series: pd.Series, parent_series: pd.Series) -> float:
    child_values = set(child_series.dropna().tolist())
    parent_values = set(parent_series.dropna().tolist())
    if not child_values:
        return 0.0
    return len(child_values.intersection(parent_values)) / len(child_values)


def _normalize_name(value: str) -> str:
    token = re.sub(r"[^a-z0-9_]", "", value.lower())
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 1:
        return token[:-1]
    return token


def _semantic_fk_score(fk_column: str, parent_name: str, parent_pk: str) -> int:
    fk_base = _normalize_name(fk_column[:-3]) if fk_column.endswith("_id") else _normalize_name(fk_column)
    parent_base = _normalize_name(parent_name)
    parent_pk_base = _normalize_name(parent_pk[:-3]) if parent_pk.endswith("_id") else _normalize_name(parent_pk)

    score = 0
    if fk_base == parent_base:
        score += 3
    if parent_pk_base and fk_base == parent_pk_base:
        score += 2
    if fk_base in parent_base or parent_base in fk_base:
        score += 1
    return score


def parse_schema(real_tables: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    if len(real_tables) < 2:
        raise ValueError("At least two tables are required for a multi-relational PoC.")

    primary_keys: Dict[str, str] = {
        table_name: _infer_primary_key(df) for table_name, df in real_tables.items()
    }

    candidate_relationships = []

    table_names = list(real_tables.keys())
    for child_name in table_names:
        child_df = real_tables[child_name]
        for column in child_df.columns:
            if column == primary_keys[child_name] or not column.endswith("_id"):
                continue

            best_match = None
            best_rank = (-1, -1.0, -1)
            for parent_name in table_names:
                if parent_name == child_name:
                    continue

                parent_pk = primary_keys[parent_name]
                ratio = _overlap_ratio(child_df[column], real_tables[parent_name][parent_pk])
                if ratio < 0.5:
                    continue

                semantic_score = _semantic_fk_score(column, parent_name, parent_pk)
                parent_row_count = len(real_tables[parent_name])
                child_row_count = len(child_df)
                size_score = 1 if parent_row_count <= child_row_count else 0
                rank = (semantic_score, ratio, size_score)

                if rank > best_rank:
                    best_rank = rank
                    best_match = parent_name

            if best_match:
                parent_pk = primary_keys[best_match]
                candidate_relationships.append(
                    {
                        "parent": best_match,
                        "parent_pk": parent_pk,
                        "child": child_name,
                        "child_fk": column,
                        "rank": best_rank,
                    }
                )

    graph = nx.DiGraph()
    graph.add_nodes_from(table_names)

    relationships: List[Tuple[str, str, str, str]] = []
    dropped_relationships: List[Tuple[str, str, str, str]] = []

    sorted_candidates = sorted(
        candidate_relationships,
        key=lambda rel: rel["rank"],
        reverse=True,
    )
    for rel in sorted_candidates:
        graph.add_edge(rel["parent"], rel["child"])
        if nx.is_directed_acyclic_graph(graph):
            relationships.append((rel["parent"], rel["parent_pk"], rel["child"], rel["child_fk"]))
        else:
            graph.remove_edge(rel["parent"], rel["child"])
            dropped_relationships.append((rel["parent"], rel["parent_pk"], rel["child"], rel["child_fk"]))

    schema_def: Dict[str, List[str]] = {table_name: [] for table_name in table_names}
    for parent, _, child, _ in relationships:
        if parent not in schema_def[child]:
            schema_def[child].append(parent)

    return {
        "primary_keys": primary_keys,
        "relationships": relationships,
        "dropped_relationships": dropped_relationships,
        "schema_def": schema_def,
        "generation_order": list(nx.topological_sort(graph)),
    }
