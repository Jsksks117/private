from __future__ import annotations

import io
import zipfile
from typing import Dict

import pandas as pd


def serialize_tables_to_zip(tables: Dict[str, pd.DataFrame]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for table_name, df in tables.items():
            zf.writestr(f"synthetic_{table_name.lower()}.csv", df.to_csv(index=False))
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
