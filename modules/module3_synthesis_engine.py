from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_integer_dtype, is_numeric_dtype
from sdv.metadata import SingleTableMetadata

from dp_ctgan import train_dp_ctgan


class DPMultiTableSynthesizer:
    def __init__(
        self,
        epsilon: float,
        db_connection_string: str = "sqlite:///poc_synth.db",
        use_full_stack: bool = True,
        epochs: int = 6,
        batch_size: int = 64,
        z_dim: int = 64,
        noise_multiplier: float = 0.8,
        max_grad_norm: float = 2.0,
        max_train_rows: int = 1200,
        max_steps_per_table: int = 10000,
        seed: int = 42,
        focus_runtime_threshold_seconds: float = 120.0,
        device: str = "cpu",
    ) -> None:
        self.epsilon = epsilon
        self.db_connection_string = db_connection_string
        self.use_full_stack = use_full_stack
        self.epochs = epochs
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.max_train_rows = max_train_rows
        self.max_steps_per_table = max_steps_per_table
        self.seed = seed
        self.focus_runtime_threshold_seconds = focus_runtime_threshold_seconds
        self.device = device

        self.last_backend_by_table: Dict[str, str] = {}
        self._elapsed_training_seconds = 0.0


    @staticmethod
    def _stable_table_seed(base_seed: int, table_name: str) -> int:
        table_hash = sum((idx + 1) * ord(ch) for idx, ch in enumerate(table_name))
        return int((base_seed + table_hash) % (2**31 - 1))


    def _build_column_info(self, model_df: pd.DataFrame) -> List[Dict[str, object]]:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(model_df)

        column_info: List[Dict[str, object]] = []
        for col in model_df.columns:
            sdtype = metadata.columns.get(col, {}).get("sdtype", "unknown")
            series = model_df[col]

            treat_as_discrete = sdtype in {"categorical", "boolean", "id"}
            if is_numeric_dtype(series) and series.nunique(dropna=True) <= 20:
                treat_as_discrete = True

            if treat_as_discrete:
                categories = list(pd.Series(series.dropna().unique()).sort_values())
                if len(categories) < 2:
                    treat_as_discrete = False
                else:
                    column_info.append(
                        {
                            "name": col,
                            "type": "discrete",
                            "output_dim": len(categories),
                            "categories": categories,
                        }
                    )

            if not treat_as_discrete:
                numeric = pd.to_numeric(series, errors="coerce")
                mean = float(numeric.mean()) if numeric.notna().any() else 0.0
                std = float(numeric.std()) if numeric.std() and not np.isnan(numeric.std()) else 1.0
                min_v = float(numeric.min()) if numeric.notna().any() else -1.0
                max_v = float(numeric.max()) if numeric.notna().any() else 1.0
                column_info.append(
                    {
                        "name": col,
                        "type": "continuous",
                        "modes": 1,
                        "mean": mean,
                        "std": std if std > 1e-8 else 1.0,
                        "min": min_v,
                        "max": max_v,
                        "is_int": bool(is_integer_dtype(series)),
                    }
                )

        return column_info

    def _encode_for_dp_ctgan(self, model_df: pd.DataFrame, column_info: List[Dict[str, object]]) -> torch.Tensor:
        n = len(model_df)
        encoded_parts = []

        for info in column_info:
            col = info["name"]
            series = model_df[col]

            if info["type"] == "discrete":
                categories = info["categories"]
                cat_to_idx = {category: idx for idx, category in enumerate(categories)}
                one_hot = np.zeros((n, info["output_dim"]), dtype=np.float32)
                mapped = [cat_to_idx.get(value, 0) for value in series.tolist()]
                one_hot[np.arange(n), mapped] = 1.0
                encoded_parts.append(one_hot)
            else:
                numeric = pd.to_numeric(series, errors="coerce").fillna(info["mean"]) 
                normalized = ((numeric - info["mean"]) / (3.0 * info["std"]))
                normalized = np.clip(normalized.to_numpy(dtype=np.float32), -1.0, 1.0).reshape(-1, 1)
                modes = np.ones((n, info["modes"]), dtype=np.float32)
                encoded_parts.append(np.concatenate([normalized, modes], axis=1))

        matrix = np.concatenate(encoded_parts, axis=1) if encoded_parts else np.zeros((n, 1), dtype=np.float32)
        return torch.tensor(matrix, dtype=torch.float32)

    def _decode_from_dp_ctgan(
        self,
        generated: torch.Tensor,
        column_info: List[Dict[str, object]],
    ) -> pd.DataFrame:
        values = generated.detach().cpu().numpy()
        decoded = pd.DataFrame()
        offset = 0

        for info in column_info:
            if info["type"] == "discrete":
                width = info["output_dim"]
                logits = values[:, offset : offset + width]
                idx = logits.argmax(axis=1)
                categories = info["categories"]
                decoded[info["name"]] = [categories[i] for i in idx]
                offset += width
            else:
                width = 1 + info["modes"]
                scalar = values[:, offset]
                restored = (scalar * (3.0 * info["std"])) + info["mean"]
                restored = np.clip(restored, info["min"], info["max"])
                if info["is_int"]:
                    restored = np.round(restored).astype(int)
                decoded[info["name"]] = restored
                offset += width

        return decoded

    def _generate_table_full(
        self,
        real_df: pd.DataFrame,
        value_columns: List[str],
        num_rows: int,
        table_epsilon: float,
        table_name: str,
        focus_table: str | None,
        allow_focus_boost: bool,
    ) -> pd.DataFrame:
        model_df = real_df[value_columns].copy()
        if model_df.empty:
            return pd.DataFrame(index=range(num_rows))

        if len(model_df) > self.max_train_rows:
            model_df = model_df.sample(n=self.max_train_rows, random_state=42)

        if len(model_df) < 8:
            raise ValueError("Not enough rows for stable DP-CTGAN training.")

        column_info = self._build_column_info(model_df)
        if not column_info:
            return pd.DataFrame(index=range(num_rows))

        train_tensor = self._encode_for_dp_ctgan(model_df, column_info)

        # Use very small batches to lower sample rate and allow many training steps within privacy budget.
        # Smaller batch_size = lower sample_rate (batch/dataset) = each step consumes less epsilon = more steps allowed
        if table_epsilon < 3.0:
            train_batch_size = max(4, len(model_df) // 50)
        elif table_epsilon < 6.0:
            train_batch_size = max(8, len(model_df) // 40)
        elif table_epsilon < 12.0:
            train_batch_size = max(12, len(model_df) // 30)
        else:
            train_batch_size = max(16, len(model_df) // 25)
        # Cap batch size to enforce small sample rate
        train_batch_size = min(train_batch_size, 48)
        if train_batch_size >= len(model_df):
            train_batch_size = max(2, len(model_df) - 1)

        effective_epochs = self.epochs
        # Increase epochs only for the table that is most likely used by TSTR.
        if allow_focus_boost and focus_table is not None and table_name == focus_table:
            if self.epsilon >= 18.0:
                effective_epochs += 8
            elif self.epsilon >= 12.0:
                effective_epochs += 5
            elif self.epsilon >= 8.0:
                effective_epochs += 2
            else:
                effective_epochs += 1
        effective_epochs = min(effective_epochs, 14)

        # Reduce noise slightly only for high-global-epsilon runs.
        effective_noise_multiplier = self.noise_multiplier
        if self.epsilon >= 18.0:
            effective_noise_multiplier = max(0.55, self.noise_multiplier - 0.15)
        elif self.epsilon >= 12.0:
            effective_noise_multiplier = max(0.65, self.noise_multiplier - 0.08)

        if allow_focus_boost and focus_table is not None and table_name == focus_table and self.epsilon >= 18.0:
            effective_noise_multiplier = max(0.50, effective_noise_multiplier - 0.05)

        table_seed = self._stable_table_seed(self.seed, table_name)

        ctgan_schema = [
            {"type": c["type"], "modes": c.get("modes", 0), "output_dim": c.get("output_dim", 0)}
            for c in column_info
        ]
        for c in ctgan_schema:
            if c["type"] == "continuous":
                c.pop("output_dim", None)
            else:
                c.pop("modes", None)

        # Debug: report table-specific training parameters so UI vs CLI runs can be compared
        print(
            f"[DP-SYNTH] table={table_name} rows={len(model_df)} eps={table_epsilon:.4f} batch={train_batch_size} epochs={effective_epochs} noise={effective_noise_multiplier:.3f} seed={table_seed}"
        )

        # Force minimum training steps so low-epsilon tables still train meaningfully
        # With smaller batch_size, this becomes achievable without exhausting budget
        min_steps = max(10, int(table_epsilon * 5))
        
        generator = train_dp_ctgan(
            real_data=train_tensor,
            column_info=ctgan_schema,
            epochs=effective_epochs,
            batch_size=train_batch_size,
            z_dim=self.z_dim,
            target_epsilon=table_epsilon,
            noise_multiplier=effective_noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            device=self.device,
            max_steps=self.max_steps_per_table,
            min_training_steps=min_steps,
            seed=table_seed,
        )

        generator.eval()
        with torch.no_grad():
            z = torch.randn(num_rows, self.z_dim, device=self.device)
            generated_tensor = generator(z)

        return self._decode_from_dp_ctgan(generated_tensor, column_info)

    def generate(
        self,
        real_tables: Dict[str, pd.DataFrame],
        schema_def: Dict[str, List[str]],
        primary_keys: Dict[str, str],
        relationships: List[Tuple[str, str, str, str]],
        num_rows: int,
        epsilon_allocation: Dict[str, float] | None = None,
        focus_table: str | None = None,
        table_row_counts: Dict[str, int] | None = None,
    ) -> Dict[str, pd.DataFrame]:
        self.last_backend_by_table = {}
        self._elapsed_training_seconds = 0.0
        fk_rng = np.random.default_rng(self.seed)

        synth_tables: Dict[str, pd.DataFrame] = {}
        for table_name in schema_def.keys():
            real_df = real_tables[table_name]
            pk_col = primary_keys[table_name]
            table_num_rows = int(table_row_counts.get(table_name, num_rows)) if table_row_counts else num_rows

            fk_cols = [r[3] for r in relationships if r[2] == table_name]
            value_cols = [c for c in real_df.columns if c != pk_col and c not in fk_cols]

            if not self.use_full_stack:
                raise RuntimeError("Strict mode requires full SDV+Opacus stack; use_full_stack=False is not allowed.")

            try:
                table_epsilon = self.epsilon
                if epsilon_allocation is not None:
                    table_epsilon = float(epsilon_allocation.get(table_name, self.epsilon))

                allow_focus_boost = self._elapsed_training_seconds < self.focus_runtime_threshold_seconds
                table_start = time.perf_counter()

                synthesized_values = self._generate_table_full(
                    real_df,
                    value_cols,
                    table_num_rows,
                    table_epsilon=max(0.05, table_epsilon),
                    table_name=table_name,
                    focus_table=focus_table,
                    allow_focus_boost=allow_focus_boost,
                )
                self._elapsed_training_seconds += time.perf_counter() - table_start
                self.last_backend_by_table[table_name] = "sdv-metadata + opacus-dp-ctgan"
            except Exception as ex:
                raise RuntimeError(
                    f"Strict full-stack synthesis failed for table '{table_name}': {ex}"
                ) from ex

            synth_df = pd.DataFrame(index=range(table_num_rows))
            for col in real_df.columns:
                if col == pk_col or col in fk_cols:
                    continue
                synth_df[col] = synthesized_values[col]

            if is_integer_dtype(real_df[pk_col]):
                synth_df[pk_col] = np.arange(1, table_num_rows + 1, dtype=int)
            else:
                synth_df[pk_col] = [f"{table_name}_{i}" for i in range(1, table_num_rows + 1)]

            for fk_col in fk_cols:
                synth_df[fk_col] = np.nan

            synth_tables[table_name] = synth_df[real_df.columns]

        for parent, parent_pk, child, child_fk in relationships:
            parent_ids = synth_tables[parent][parent_pk].values
            child_size = len(synth_tables[child])
            synth_tables[child][child_fk] = fk_rng.choice(parent_ids, size=child_size, replace=True)

        return synth_tables
