from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal

_DEFAULT_PERCENTILES: tuple[Decimal, ...] = (
    Decimal("25"),
    Decimal("50"),
    Decimal("75"),
    Decimal("90"),
    Decimal("95"),
    Decimal("99"),
)

_Z_SCORES: dict[Decimal, Decimal] = {
    Decimal("0.80"): Decimal("1.2815515655446004"),
    Decimal("0.90"): Decimal("1.6448536269514722"),
    Decimal("0.95"): Decimal("1.959963984540054"),
    Decimal("0.98"): Decimal("2.3263478740408408"),
    Decimal("0.99"): Decimal("2.5758293035489004"),
}


def _to_decimal(value: int | float | str | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _as_sorted_decimals(values: Iterable[int | float | str | Decimal]) -> list[Decimal]:
    decimals = sorted(_to_decimal(value) for value in values)
    if not decimals:
        raise ValueError("values must not be empty")
    return decimals


def median(values: Sequence[int | float | str | Decimal]) -> Decimal:
    """Return median value using Decimal arithmetic.

    Args:
        values: Numeric sequence with at least one element.

    Returns:
        Median as Decimal. For even-length sequences, returns average of two middle values.

    Raises:
        ValueError: If ``values`` is empty.

    Edge cases:
        - Single value: returns that value.
        - All same values: returns the repeated value.

    Example:
        >>> median([1, 2, 3, 4])
        Decimal('2.5')
    """

    sorted_values = _as_sorted_decimals(values)
    length = len(sorted_values)
    mid = length // 2
    if length % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / Decimal("2")


def weighted_mean(
    values: Sequence[int | float | str | Decimal],
    weights: Sequence[int | float | str | Decimal],
) -> Decimal:
    """Return weighted average of values.

    Args:
        values: Numeric values.
        weights: Numeric weights for each value.

    Returns:
        Weighted mean as Decimal.

    Raises:
        ValueError: If input lengths differ, inputs are empty, weights contain negative values,
            or sum of weights is zero.

    Edge cases:
        - Single value: returns that value when corresponding weight is positive.
        - All same values: returns that same value.

    Example:
        >>> weighted_mean([10, 20], [1, 3])
        Decimal('17.5')
    """

    if not values:
        raise ValueError("values must not be empty")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")

    decimal_values = [_to_decimal(value) for value in values]
    decimal_weights = [_to_decimal(weight) for weight in weights]

    if any(weight < 0 for weight in decimal_weights):
        raise ValueError("weights must be non-negative")

    weight_sum = sum(decimal_weights, Decimal("0"))
    if weight_sum == 0:
        raise ValueError("sum of weights must be greater than zero")

    weighted_total = sum(
        value * weight for value, weight in zip(decimal_values, decimal_weights, strict=True)
    )
    return weighted_total / weight_sum


def percentile(
    values: Sequence[int | float | str | Decimal], p: int | float | str | Decimal
) -> Decimal:
    """Return the p-th percentile via linear interpolation.

    Args:
        values: Numeric sequence with at least one element.
        p: Percentile in inclusive range [0, 100].

    Returns:
        Percentile value as Decimal.

    Raises:
        ValueError: If ``values`` is empty or ``p`` is outside [0, 100].

    Edge cases:
        - Single value: always returns the single value.
        - All same values: returns that repeated value.

    Example:
        >>> percentile([1, 2, 3, 4, 5], 90)
        Decimal('4.6')
    """

    sorted_values = _as_sorted_decimals(values)
    p_decimal = _to_decimal(p)
    if p_decimal < 0 or p_decimal > 100:
        raise ValueError("p must be between 0 and 100 inclusive")

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (Decimal(len(sorted_values) - 1) * p_decimal) / Decimal("100")
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - Decimal(lower_index)

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + ((upper_value - lower_value) * fraction)


def percentiles(
    values: Sequence[int | float | str | Decimal],
    ps: Sequence[int | float | str | Decimal] | None = None,
) -> dict[Decimal, Decimal]:
    """Return multiple percentile values for a dataset.

    Args:
        values: Numeric sequence with at least one element.
        ps: Requested percentile points. Defaults to (25, 50, 75, 90, 95, 99).

    Returns:
        Mapping of percentile point to percentile value.

    Raises:
        ValueError: If ``values`` is empty or any percentile is outside [0, 100].

    Example:
        >>> percentiles([1, 2, 3, 4, 5], [25, 50, 75])
        {Decimal('25'): Decimal('2'), Decimal('50'): Decimal('3'), Decimal('75'): Decimal('4')}
    """

    requested = tuple(_DEFAULT_PERCENTILES if ps is None else (_to_decimal(p) for p in ps))
    if not requested:
        raise ValueError("ps must not be empty")
    return {p: percentile(values, p) for p in requested}


def iqr(values: Sequence[int | float | str | Decimal]) -> Decimal:
    """Return interquartile range (Q3 - Q1).

    Args:
        values: Numeric sequence with at least one element.

    Returns:
        Interquartile range as Decimal.

    Raises:
        ValueError: If ``values`` is empty.

    Edge cases:
        - Single value: returns Decimal('0').
        - All same values: returns Decimal('0').

    Example:
        >>> iqr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        Decimal('4.5')
    """

    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return Decimal("0")
    q1 = percentile(values, Decimal("25"))
    q3 = percentile(values, Decimal("75"))
    return q3 - q1


def mad(values: Sequence[int | float | str | Decimal]) -> Decimal:
    """Return median absolute deviation (MAD).

    Args:
        values: Numeric sequence with at least one element.

    Returns:
        Median absolute deviation as Decimal.

    Raises:
        ValueError: If ``values`` is empty.

    Edge cases:
        - Single value: returns Decimal('0').
        - All same values: returns Decimal('0').

    Example:
        >>> mad([10, 10, 11, 12, 100])
        Decimal('1')
    """

    sorted_values = _as_sorted_decimals(values)
    if len(sorted_values) == 1:
        return Decimal("0")
    center = median(sorted_values)
    absolute_deviations = [abs(value - center) for value in sorted_values]
    return median(absolute_deviations)


def confidence_interval(
    values: Sequence[int | float | str | Decimal],
    confidence: int | float | str | Decimal = "0.95",
) -> tuple[Decimal, Decimal]:
    """Return two-sided confidence interval for the sample mean.

    Args:
        values: Numeric sequence with at least one element.
        confidence: Confidence level in (0, 1), supported: 0.80, 0.90, 0.95, 0.98, 0.99.

    Returns:
        Tuple ``(lower_bound, upper_bound)`` as Decimals.

    Raises:
        ValueError: If ``values`` is empty, confidence is unsupported, or confidence outside (0, 1).

    Edge cases:
        - Single value: returns ``(value, value)``.
        - All same values: returns ``(mean, mean)``.
        - Statistical validity note: results with n < 5 are mathematically valid but lower-confidence.

    Example:
        >>> confidence_interval([10, 12, 11, 13, 12], Decimal('0.95'))
        (Decimal('10.614...'), Decimal('12.585...'))
    """

    decimal_values = _as_sorted_decimals(values)
    n = len(decimal_values)
    if n == 1:
        value = decimal_values[0]
        return value, value

    confidence_level = _to_decimal(confidence)
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence must be between 0 and 1 (exclusive)")
    z_score = _Z_SCORES.get(confidence_level)
    if z_score is None:
        raise ValueError("unsupported confidence level; supported: 0.80, 0.90, 0.95, 0.98, 0.99")

    n_decimal = Decimal(n)
    mean_value = sum(decimal_values, Decimal("0")) / n_decimal
    squared_diff_sum = sum((value - mean_value) ** 2 for value in decimal_values)
    sample_variance = squared_diff_sum / Decimal(n - 1)

    if sample_variance == 0:
        return mean_value, mean_value

    standard_error = (sample_variance / n_decimal).sqrt()
    margin = z_score * standard_error
    return mean_value - margin, mean_value + margin


def _sql_where(where_clause: str) -> str:
    trimmed = where_clause.strip()
    if not trimmed:
        return ""
    if trimmed.lower().startswith("where "):
        return f" {trimmed}"
    return f" WHERE {trimmed}"


def clickhouse_median(table: str, column: str, where_clause: str = "") -> str:
    """Return ClickHouse SQL for median aggregate.

    Args:
        table: Table name.
        column: Numeric column to aggregate.
        where_clause: Optional filter expression, with or without ``WHERE`` prefix.

    Returns:
        SQL query string using ClickHouse ``median()``.

    Example:
        >>> clickhouse_median("contracts", "contract_sum", "year = 2025")
        'SELECT median(contract_sum) AS median_value FROM contracts WHERE year = 2025'
    """

    return f"SELECT median({column}) AS median_value FROM {table}{_sql_where(where_clause)}"


def clickhouse_percentile(
    table: str,
    column: str,
    percentile_value: int | float | str | Decimal,
    where_clause: str = "",
) -> str:
    """Return ClickHouse SQL for a single quantile aggregate.

    Args:
        table: Table name.
        column: Numeric column to aggregate.
        percentile_value: Percentile in [0, 100].
        where_clause: Optional filter expression, with or without ``WHERE`` prefix.

    Returns:
        SQL query string using ClickHouse ``quantile(level)``.

    Raises:
        ValueError: If percentile is outside [0, 100].

    Example:
        >>> clickhouse_percentile("lots", "amount", 95)
        'SELECT quantile(0.95)(amount) AS percentile_value FROM lots'
    """

    p = _to_decimal(percentile_value)
    if p < 0 or p > 100:
        raise ValueError("percentile must be between 0 and 100 inclusive")
    level = p / Decimal("100")
    return f"SELECT quantile({level})(%s) AS percentile_value FROM %s%s" % (
        column,
        table,
        _sql_where(where_clause),
    )


def clickhouse_quantiles(
    table: str,
    column: str,
    quantiles: Sequence[int | float | str | Decimal],
    where_clause: str = "",
) -> str:
    """Return ClickHouse SQL for multiple quantile aggregates.

    Args:
        table: Table name.
        column: Numeric column to aggregate.
        quantiles: Percentile points in [0, 100].
        where_clause: Optional filter expression, with or without ``WHERE`` prefix.

    Returns:
        SQL query string using ClickHouse ``quantiles(...)``.

    Raises:
        ValueError: If ``quantiles`` is empty or contains values outside [0, 100].

    Example:
        >>> clickhouse_quantiles("contracts", "contract_sum", [25, 50, 75])
        'SELECT quantiles(0.25, 0.5, 0.75)(contract_sum) AS quantiles_values FROM contracts'
    """

    if not quantiles:
        raise ValueError("quantiles must not be empty")

    levels: list[str] = []
    for quantile_value in quantiles:
        q = _to_decimal(quantile_value)
        if q < 0 or q > 100:
            raise ValueError("all quantiles must be between 0 and 100 inclusive")
        levels.append(str(q / Decimal("100")))

    levels_csv = ", ".join(levels)
    return f"SELECT quantiles({levels_csv})(%s) AS quantiles_values FROM %s%s" % (
        column,
        table,
        _sql_where(where_clause),
    )
