"""GoDe-style pruning schedule helpers used in NiFi Artifact Synthesis."""

from __future__ import annotations

import math
from typing import List


DEFAULT_GODE_MIN_CARDINALITY = 4096


def compute_gode_pruning_cardinalities(
    minimum_cardinality: int,
    maximum_cardinality: int,
    num_levels: int = 3,
) -> List[int]:
    """Compute the paper's logarithmic pruning schedule for ``|G_l|``.

    The paper defines three pruning levels ``{G_0, G_1, G_2}`` with:
    ``|G_0| = c_min`` and logarithmic interpolation up to ``|G_{L-1}|``.

    Args:
        minimum_cardinality: ``c_min`` from the paper.
        maximum_cardinality: ``|G_{L-1}|`` from the paper.
        num_levels: Number of levels ``L``. NiFi uses ``L=3``.

    Returns:
        Monotonic integer cardinalities for ``|G_0| ... |G_{L-1}|``.
    """
    if num_levels < 2:
        raise ValueError("num_levels must be >= 2")
    if minimum_cardinality <= 0:
        raise ValueError("minimum_cardinality must be > 0")
    if maximum_cardinality < minimum_cardinality:
        raise ValueError("maximum_cardinality must be >= minimum_cardinality")

    log_min = math.log(float(minimum_cardinality))
    log_max = math.log(float(maximum_cardinality))

    levels: List[int] = []
    denominator = float(num_levels - 1)
    for level_idx in range(num_levels):
        ratio = float(level_idx) / denominator
        cardinality = math.exp(log_min + ratio * (log_max - log_min))
        levels.append(int(round(cardinality)))

    levels[0] = int(minimum_cardinality)
    levels[-1] = int(maximum_cardinality)

    # Guard against repeated values due to integer rounding.
    for idx in range(1, len(levels)):
        levels[idx] = max(levels[idx], levels[idx - 1])

    return levels
