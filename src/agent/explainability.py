"""Explainability utilities for agent response generation.

Provides functions for:
- Calculating confidence scores based on multiple signals (sample size, retrieval quality,
  data completeness, query type, data quality)
- Generating data lineage information (contracts, ENSTR categories, entity types)
- Formatting methodology descriptions
- Generating limitation statements based on data quality
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict


class DataLineage(TypedDict):
    """Data lineage information for response generation."""

    contracts: list[int]
    enstr_categories: list[str]
    entity_types: dict[str, int]
    date_range: dict[str, str]


# Supported query types for confidence calculation
SupportedQueryType = Literal["SEARCH", "COMPARISON", "ANALYTICS", "ANOMALY_DETECTION", "FAIRNESS"]

# Quality scores from data quality module
QualityScore = Literal["PASS", "WARN", "FAIL"]

# Query type reliability factors (higher = more reliable)
QUERY_TYPE_FACTORS: dict[SupportedQueryType, float] = {
    "SEARCH": 0.10,
    "COMPARISON": 0.08,
    "ANALYTICS": 0.08,
    "ANOMALY_DETECTION": 0.05,  # Higher uncertainty due to outlier detection
    "FAIRNESS": 0.05,  # Depends on finding enough similar contracts
}

# Data quality penalties
QUALITY_PENALTIES: dict[QualityScore, float] = {
    "PASS": 0.0,
    "WARN": -0.10,
    "FAIL": -0.25,
}


def calculate_confidence(
    sample_size: int,
    data_completeness: float = 1.0,
    has_entities: bool = False,
    avg_retrieval_score: float = 0.5,
    query_type: SupportedQueryType = "SEARCH",
    quality_score: QualityScore = "PASS",
    outlier_ratio: float = 0.0,
) -> float:
    """Calculate confidence score based on multiple signals.

    This function implements a multi-factor confidence calculation that aligns with
    the Terms of Reference requirement for "data quality assessment and risks".

    Factors (in order of weight):
    1. Sample size factor (base confidence from n)
    2. Retrieval quality factor (from RAG scores)
    3. Data completeness factor (from field presence)
    4. Query type reliability factor
    5. Data quality penalty (from data quality module)
    6. Statistical stability factor (from outlier ratio)
    7. Entity extraction boost

    Args:
        sample_size: Number of records analyzed
        data_completeness: Proportion of records with complete data (0.0-1.0)
        has_entities: Whether entity extraction was successful
        avg_retrieval_score: Average retrieval score from RAG (0.0-1.0)
        query_type: Type of query for reliability factor
        quality_score: Overall data quality (PASS/WARN/FAIL)
        outlier_ratio: Proportion of outliers in data (affects stability)

    Returns:
        Confidence score from 0.0 to 1.0 (rounded to 3 decimal places)
    """
    if sample_size == 0:
        return 0.0

    # 1. Sample size factor (more granular than previous implementation)
    if sample_size < 5:
        n_factor = 0.20
    elif sample_size < 10:
        n_factor = 0.35
    elif sample_size < 20:
        n_factor = 0.50
    elif sample_size < 50:
        n_factor = 0.60
    elif sample_size < 100:
        n_factor = 0.65
    else:
        n_factor = 0.70

    # 2. Retrieval quality factor (from RAG scores) - max 8%
    retrieval_factor = max(0.0, min(1.0, avg_retrieval_score)) * 0.08

    # 3. Data completeness factor - max 8%
    completeness_factor = max(0.0, min(1.0, data_completeness)) * 0.08

    # 4. Query type reliability factor
    query_factor = QUERY_TYPE_FACTORS.get(query_type, 0.05)

    # 5. Data quality penalty
    quality_penalty = QUALITY_PENALTIES.get(quality_score, 0.0)

    # 6. Statistical stability factor (high outlier ratio = less confidence)
    stability_factor = -0.05 * min(1.0, outlier_ratio * 10)

    # 7. Entity extraction boost
    entity_boost = 0.03 if has_entities else 0.0

    # Combine all factors (max ~0.98)
    confidence = (
        n_factor
        + retrieval_factor
        + completeness_factor
        + query_factor
        + quality_penalty
        + stability_factor
        + entity_boost
    )

    return round(max(0.0, min(1.0, confidence)), 3)


# Keep backward-compatible alias for existing code
def calculate_confidence_legacy(
    sample_size: int,
    data_completeness: float = 1.0,
    has_entities: bool = False,
) -> float:
    """Legacy confidence calculation (backward compatible).

    This maintains backward compatibility with existing code that doesn't
    provide the new parameters.

    Args:
        sample_size: Number of records analyzed
        data_completeness: Proportion of records with complete data (0.0-1.0)
        has_entities: Whether entity extraction was successful

    Returns:
        Confidence score from 0.0 to 1.0
    """
    if sample_size == 0:
        return 0.0

    # Base confidence by sample size
    if sample_size < 5:
        base = 0.3
    elif sample_size < 10:
        base = 0.5
    elif sample_size < 20:
        base = 0.7
    else:
        base = 0.85

    # Entity extraction boost
    entity_boost = 0.05 if has_entities else 0.0

    # Clamp data_completeness to [0.0, 1.0] and apply
    completeness_factor = max(0.0, min(1.0, data_completeness))
    confidence = min(1.0, base + entity_boost) * completeness_factor

    return round(confidence, 3)


def generate_data_lineage(
    chunks: Sequence[Mapping[str, object]] | None = None,
    analytics_results: Mapping[str, object] | None = None,
) -> DataLineage:
    """Generate data lineage information from retrieved chunks and analytics.

    Extracts:
    - Contract/procurement IDs analyzed
    - ENSTR classification codes used
    - Entity types (contracts, lots, plans, announcements)
    - Date range of data

    Args:
        chunks: Retrieved document chunks with metadata
        analytics_results: Analytics query results

    Returns:
        DataLineage dict with contracts, ENSTR categories, entity types, and date range
    """
    contracts: list[int] = []
    enstr_categories_set: set[str] = set()
    entity_types: dict[str, int] = {}
    date_range: dict[str, str] = {"from": "", "to": ""}

    if chunks:
        for chunk in chunks:
            if isinstance(chunk, Mapping):
                # Extract procurement ID
                procurement_id = chunk.get("procurement_id", 0)
                if isinstance(procurement_id, int) and procurement_id > 0:
                    contracts.append(procurement_id)

                # Extract entity type
                entity_type = str(chunk.get("entity_type", "unknown")).lower()
                if entity_type != "unknown":
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

                # Extract dates from metadata
                metadata = chunk.get("metadata")
                if isinstance(metadata, Mapping):
                    sign_date = metadata.get("sign_date")
                    if sign_date:
                        sign_date_str = str(sign_date)
                        if not date_range["from"] or sign_date_str < date_range["from"]:
                            date_range["from"] = sign_date_str
                        if not date_range["to"] or sign_date_str > date_range["to"]:
                            date_range["to"] = sign_date_str

    # Extract ENSTR categories from analytics if available
    if analytics_results and isinstance(analytics_results, Mapping):
        spend_by_enstr = analytics_results.get("spend_by_enstr", [])
        if isinstance(spend_by_enstr, Sequence):
            for row in spend_by_enstr:
                if isinstance(row, Sequence) and len(row) > 0:
                    enstr_code = str(row[0])
                    if enstr_code and enstr_code != "None":
                        enstr_categories_set.add(enstr_code)

    return {
        "contracts": list(dict.fromkeys(contracts)),  # Remove duplicates, preserve order
        "enstr_categories": sorted(list(enstr_categories_set)),
        "entity_types": entity_types,
        "date_range": date_range,
    }


def format_methodology(query_type: str, methods_used: list[str] | None = None) -> str:
    """Format methodology description for response.

    Args:
        query_type: Type of query (SEARCH, COMPARISON, ANALYTICS, ANOMALY_DETECTION, FAIRNESS)
        methods_used: List of specific methods applied

    Returns:
        Human-readable methodology description
    """
    if methods_used is None:
        methods_used = []

    # Default methodologies by query type
    default_methods: dict[str, str] = {
        "SEARCH": "hybrid retrieval (vector + BM25) + reranking + relevance scoring",
        "COMPARISON": "weighted averages + median statistics + outlier detection (z > 2sd)",
        "ANALYTICS": ("aggregate time-series analysis + year-over-year trends + segment breakdown"),
        "ANOMALY_DETECTION": (
            "statistical deviation scoring + z-score analysis + baseline comparison"
        ),
        "FAIRNESS": "relative fairness scoring + median group comparison + equity metrics",
    }

    base_methodology = default_methods.get(
        query_type, "deterministic scoring + baseline comparison"
    )

    if methods_used:
        return f"{base_methodology} + {' + '.join(methods_used)}"
    return base_methodology


def generate_limitations(
    sample_size: int,
    data_completeness: float,
    temporal_coverage: str = "full period",
) -> list[str]:
    """Generate list of limitation statements based on data quality.

    Args:
        sample_size: Number of records analyzed
        data_completeness: Proportion of complete records (0.0-1.0)
        temporal_coverage: Description of time period covered

    Returns:
        List of limitation statements
    """
    limitations: list[str] = []

    # Sample size limitations
    if sample_size == 0:
        limitations.append(
            "No data matched the query filters; no statistical conclusion can be drawn."
        )
    elif sample_size < 5:
        limitations.append(
            f"Small sample size (n={sample_size}) significantly limits statistical confidence."
        )
    elif sample_size < 10:
        limitations.append(
            f"Moderate sample size (n={sample_size}) may affect statistical inference confidence."
        )
    elif sample_size < 5:
        limitations.append(
            f"Small sample size (n={sample_size}) significantly limits statistical confidence."
        )
    elif sample_size < 10:
        msg = f"Moderate sample size (n={sample_size}); may affect statistical inference."
        limitations.append(msg)

    # Data completeness limitations
    if data_completeness < 0.5:
        limitations.append(
            f"Significant missing data fields ({100 - data_completeness * 100:.0f}% incomplete); "
            "results may be unreliable."
        )
    elif data_completeness < 0.8:
        limitations.append(
            "Some records have missing key fields (e.g., sign_date, contract_sum); "
            "analysis is partial."
        )

    # Temporal coverage limitations
    if temporal_coverage == "partial period":
        limitations.append(
            "Analysis covers partial time period; trends may not reflect full context."
        )

    # Default limitation if none found
    if not limitations:
        limitations.append("No critical data quality constraints detected in retrieved sample.")

    return limitations


def extract_enstr_codes(chunks: Sequence[Mapping[str, object]]) -> list[str]:
    """Extract unique ENSTR classification codes from document metadata.

    Args:
        chunks: Retrieved document chunks with metadata

    Returns:
        Sorted list of unique ENSTR codes
    """
    codes: set[str] = set()

    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue

        metadata = chunk.get("metadata")
        if not isinstance(metadata, Mapping):
            continue

        enstr_code = metadata.get("enstr_code")
        if enstr_code:
            code_str = str(enstr_code).strip()
            if code_str:
                codes.add(code_str)

    return sorted(list(codes))


def count_entity_types(chunks: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Count occurrences of each entity type in chunks.

    Args:
        chunks: Retrieved document chunks with metadata

    Returns:
        Dictionary mapping entity type to count
    """
    counts: dict[str, int] = {}

    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue

        entity_type = str(chunk.get("entity_type", "unknown")).lower()
        counts[entity_type] = counts.get(entity_type, 0) + 1

    return counts
