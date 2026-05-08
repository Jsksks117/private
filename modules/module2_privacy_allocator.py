from __future__ import annotations

from typing import Dict, List

from schema_budget_allocator import SchemaBudgetAllocator


def allocate_privacy(schema_def: Dict[str, List[str]], total_epsilon: float) -> Dict[str, object]:
    allocator = SchemaBudgetAllocator(schema=schema_def, total_epsilon=total_epsilon)
    return allocator.run()
