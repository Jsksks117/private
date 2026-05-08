import os
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from modules.module1_schema_parser import parse_schema
from modules.module2_privacy_allocator import allocate_privacy
from modules.module3_synthesis_engine import DPMultiTableSynthesizer
from modules.module4_output_serializer import serialize_tables_to_zip
from modules.module5_evaluation_engine import evaluate_synthetic_quality
from modules.repro_benchmark import run_repro_benchmark

plt.style.use("dark_background")
sns.set_style("darkgrid", {"axes.facecolor": ".1", "grid.color": ".2"})

st.set_page_config(page_title="Graph-Coordinated DP Synthesizer (PoC)", page_icon="🌌", layout="wide")


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #e2e8f0;
            font-family: 'Inter', sans-serif;
        }
        .main-title {
            background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 42px;
            font-weight: 900;
            margin-bottom: 5px;
            text-align: center;
        }
        .sub-title {
            color: #94a3b8;
            font-size: 17px;
            text-align: center;
            margin-bottom: 28px;
            font-weight: 300;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            color: #ffffff;
            font-weight: 700;
            border-radius: 12px;
            border: none;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _normalize_table_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    base = re.sub(r"^confidential_", "", base, flags=re.IGNORECASE)
    return base.replace("-", "_").replace(" ", "_").title()


def _load_uploaded_tables(uploaded_files) -> dict:
    tables = {}
    for file in uploaded_files:
        table_name = _normalize_table_name(file.name)
        tables[table_name] = pd.read_csv(file)
    return tables


def _pick_tstr_focus_table(real_tables: dict) -> str | None:
    preferred = ["is_delayed", "fraud_flag", "is_priority", "oversize_flag", "fare_class", "loyalty_tier"]
    for table_name, df in real_tables.items():
        for col in preferred:
            if col in df.columns:
                unique_count = df[col].dropna().nunique()
                if 2 <= unique_count <= 10:
                    return table_name
    return None


inject_custom_css()

st.markdown('<p class="main-title">🔒 Graph-Coordinated Differentially Private Multi-Relational Data Synthesis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">PoC with modular architecture: parse schema, allocate epsilon on DAG, synthesize, serialize, and evaluate.</p>', unsafe_allow_html=True)

col_config, col_main = st.columns([1, 2])

with col_config:
    st.subheader("⚙️ Input and Schema Parsing")
    uploaded_files = st.file_uploader(
        "Upload 2 or more table CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    epsilon = st.slider(
        "Global privacy budget (epsilon)",
        min_value=1.0,
        max_value=20.0,
        value=6.0,
        step=0.5,
    )
    num_rows = st.number_input("Rows to generate per table", min_value=50, max_value=5000, value=300, step=50)

    can_run = uploaded_files is not None and len(uploaded_files) >= 2
    if not can_run:
        st.info("Upload at least two CSV files to start the PoC pipeline.")
        run_button = False
    else:
        run_button = st.button("🚀 Run PoC Pipeline", use_container_width=True)

with col_main:
    st.subheader("📊 Pipeline Output")
    if not run_button:
        st.info("Run the pipeline to see schema parsing, DAG allocation, synthesis, and evaluation outputs.")
    else:
        real_tables = _load_uploaded_tables(uploaded_files)

        with st.status("Parsing schema and inferring relationships", expanded=True) as status1:
            schema_info = parse_schema(real_tables)

            rel_df = pd.DataFrame(
                schema_info["relationships"],
                columns=["Parent Table", "Parent PK", "Child Table", "Child FK"],
            )
            st.write("Detected Primary Keys")
            st.dataframe(
                pd.DataFrame(
                    [{"Table": t, "Primary Key": pk} for t, pk in schema_info["primary_keys"].items()]
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.write("Detected Relationships")
            if rel_df.empty:
                st.warning("No foreign key relationships detected by heuristics. Upload linked tables with *_id columns.")
            else:
                st.dataframe(rel_df, hide_index=True, use_container_width=True)
            status1.update(label="Schema parsing complete", state="complete", expanded=False)

        with st.status("Allocating privacy budget on DAG", expanded=True) as status2:
            alloc = allocate_privacy(schema_info["schema_def"], epsilon)
            total_allocated = sum(alloc["epsilon_allocation"].values())
            alloc_rows = [
                {
                    "Table": table,
                    "Downstream Fan-Out": alloc["fanout_counts"][table],
                    "Allocated Epsilon": round(alloc["epsilon_allocation"][table], 4),
                    "% of Budget": round((alloc["epsilon_allocation"][table] / total_allocated) * 100, 2),
                }
                for table in alloc["generation_order"]
            ]
            st.dataframe(pd.DataFrame(alloc_rows), hide_index=True, use_container_width=True)
            status2.update(label="Privacy allocation complete", state="complete", expanded=False)

        with st.status("Synthesizing tables and packaging outputs", expanded=True) as status3:
            synthesizer = DPMultiTableSynthesizer(
                epsilon=epsilon,
                db_connection_string="sqlite:///poc_modular.db",
                seed=42,
                focus_runtime_threshold_seconds=90.0,
            )
            focus_table = _pick_tstr_focus_table(real_tables)
            table_row_counts = {table_name: len(df) for table_name, df in real_tables.items()}
            synth_tables = synthesizer.generate(
                real_tables=real_tables,
                schema_def=schema_info["schema_def"],
                primary_keys=schema_info["primary_keys"],
                relationships=schema_info["relationships"],
                num_rows=int(num_rows),
                epsilon_allocation=alloc["epsilon_allocation"],
                focus_table=focus_table,
                table_row_counts=table_row_counts,
            )
            zip_bytes = serialize_tables_to_zip(synth_tables)

            st.write("Synthesis backend per table")
            st.json(synthesizer.last_backend_by_table)
            st.caption(f"Repro config: seed=42, focus_runtime_threshold_seconds=90.0, focus_table={focus_table}")
            st.info("Strict mode is active: full SDV+Opacus synthesis is required for every table.")
            status3.update(label="Synthesis and serialization complete", state="complete", expanded=False)

        st.write("### Synthetic Table Previews")
        tabs = st.tabs(list(synth_tables.keys()))
        for tab, table_name in zip(tabs, synth_tables.keys()):
            with tab:
                st.dataframe(synth_tables[table_name].head(10), use_container_width=True)
                st.download_button(
                    label=f"Download {table_name} CSV",
                    data=synth_tables[table_name].to_csv(index=False),
                    file_name=f"synthetic_{table_name.lower()}.csv",
                    mime="text/csv",
                )

        st.download_button(
            label="Download All Synthetic Tables (ZIP)",
            data=zip_bytes,
            file_name="synthetic_database_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )

        with st.status("Evaluating fidelity, integrity, utility, and privacy", expanded=True) as status5:
            metrics = evaluate_synthetic_quality(
                real_tables=real_tables,
                synth_tables=synth_tables,
                relationships=schema_info["relationships"],
            )

            avg_ks = metrics["summary"].get("average_ks_score", 0.0)
            avg_card_error = metrics["summary"].get("average_cardinality_error", 0.0)
            k1, k2, k3 = st.columns(3)
            k1.metric("Average KS (lower is better)", f"{avg_ks:.3f}")

            tstr_acc = metrics["tstr"].get("TSTR_Accuracy") if metrics["tstr"] else None
            tstr_auc = metrics["tstr"].get("TSTR_AUC") if metrics["tstr"] else None
            k2.metric("TSTR Accuracy", f"{tstr_acc * 100:.1f}%" if tstr_acc is not None else "N/A")
            k3.metric("TSTR AUC", f"{tstr_auc:.3f}" if tstr_auc is not None else "N/A")
            st.metric("Average Cardinality Error (lower is better)", f"{avg_card_error:.3f}")
            st.caption(
                f"TSTR target used: table={metrics.get('target_table')} column={metrics.get('target_col')}"
            )
            if metrics.get("tstr_error"):
                st.warning(f"TSTR could not be computed: {metrics['tstr_error']}")

            # One-click reproducible benchmark
            if st.button("Run Repro Benchmark (low=2.0 vs high=20.0)", use_container_width=True):
                with st.spinner("Running reproducible benchmark (this may take a minute)..."):
                    try:
                        bench = run_repro_benchmark(real_tables, num_rows=int(num_rows), low_eps=2.0, high_eps=20.0, focus_table=focus_table, seed=42)
                        df = pd.DataFrame([bench['low'], bench['high']])
                        df = df.set_index('eps')
                        st.write("### Repro Benchmark Results")
                        st.dataframe(df)
                    except Exception as ex:
                        st.error(f"Repro benchmark failed: {ex}")

            st.write("Relational Integrity Details")
            st.json(metrics["relational"])
            status5.update(label="Evaluation complete", state="complete")

        # Build numeric candidates by table for balanced selection
        numeric_by_table = {}
        for table_name, real_df in real_tables.items():
            numeric_by_table[table_name] = []
            for col in real_df.columns:
                if pd.api.types.is_numeric_dtype(real_df[col]) and col != schema_info["primary_keys"].get(table_name):
                    numeric_by_table[table_name].append(col)
        
        # Smart selection: prioritize one from each table, then fill remaining slots
        selected = []
        for table_name in numeric_by_table:
            if numeric_by_table[table_name] and len(selected) < 6:
                selected.append((table_name, numeric_by_table[table_name][0]))
        
        # Fill remaining slots with additional columns from tables with multiple numeric columns
        for table_name in numeric_by_table:
            for col in numeric_by_table[table_name][1:]:
                if len(selected) < 6:
                    selected.append((table_name, col))
        
        if selected:
            st.write("### Distribution Comparisons")
            n = len(selected)
            ncols = min(3, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
            if isinstance(axes, np.ndarray):
                axes = axes.flatten().tolist()
            else:
                axes = [axes]

            for idx, (ax, (table_name, col)) in enumerate(zip(axes, selected)):
                sns.kdeplot(real_tables[table_name][col].dropna(), fill=True, label="Real", ax=ax, color="#38bdf8")
                sns.kdeplot(synth_tables[table_name][col].dropna(), fill=True, label="Synthetic", ax=ax, color="#c084fc")
                ax.set_title(f"{table_name}.{col}", color="white")
                # Only show legend on first plot to avoid duplication
                if idx == 0:
                    ax.legend(loc="upper right")
                else:
                    ax.legend().set_visible(False)

            for ax in axes[len(selected):]:
                ax.axis("off")

            plt.tight_layout()
            st.pyplot(fig)
    
