from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UncertaintyThresholds:
    dangerous_mid_min: float
    dangerous_mid_max: float
    gap_abs_max: float
    smoke_dominance_min: float
    controlled_dominance_min: float
    high_flicker_min: float
    moderate_spread_min: float
    moderate_spread_max: float


@dataclass
class LocalScoreWeights:
    dangerous_fire_index: float
    spread_normalized: float
    smoke: float
    fire_controlled_gap_positive: float
    fire_controlled_ratio_capped: float
    flicker_penalty: float


@dataclass
class ScenarioThresholds:
    emergency_enter: float
    emergency_exit: float
    hazard_enter: float
    hazard_exit: float
    elevated_enter: float
    no_fire_risk_max: float
    elevated_max_fire_conf: float
    elevated_min_smoke_conf: float


def is_uncertain(interval_metrics: dict, thresholds: UncertaintyThresholds) -> bool:
    agg = interval_metrics["aggregate_relative_confidence"]
    risk = interval_metrics["risk_numbers"]

    dangerous = risk["dangerous_fire_index"]
    gap = abs(risk["fire_vs_controlled_gap"])
    smoke = agg["smoke"]
    controlled = agg["controlled_fire"]
    flicker = risk["flicker_normalized"]
    spread = risk["spread_normalized"]

    mid_danger = thresholds.dangerous_mid_min <= dangerous <= thresholds.dangerous_mid_max
    near_tie = gap < thresholds.gap_abs_max
    smoke_but_controlled = smoke >= thresholds.smoke_dominance_min and controlled >= thresholds.controlled_dominance_min
    flicker_spread_conflict = (
        flicker >= thresholds.high_flicker_min
        and thresholds.moderate_spread_min <= spread <= thresholds.moderate_spread_max
    )

    return mid_danger or near_tie or smoke_but_controlled or flicker_spread_conflict


def compute_local_score(aggregate_relative_confidence: dict, risk_numbers: dict, weights: LocalScoreWeights) -> float:
    fire = aggregate_relative_confidence["fire"]
    smoke = aggregate_relative_confidence["smoke"]
    ratio = min(risk_numbers["fire_to_controlled_ratio"], 5.0) / 5.0
    gap_pos = max(risk_numbers["fire_vs_controlled_gap"], 0.0)

    score = (
        weights.dangerous_fire_index * risk_numbers["dangerous_fire_index"]
        + weights.spread_normalized * risk_numbers["spread_normalized"]
        + weights.smoke * smoke
        + weights.fire_controlled_gap_positive * gap_pos
        + weights.fire_controlled_ratio_capped * ratio
        - weights.flicker_penalty * risk_numbers["flicker_normalized"] * max(0.0, 1.0 - fire)
    )
    return max(0.0, min(1.0, score))


def compute_decision_confidence(
    aggregate_relative_confidence: dict,
    risk_numbers: dict,
    openai_output_optional: dict | None,
) -> float:
    ordered = sorted(aggregate_relative_confidence.values(), reverse=True)
    dominance_gap = ordered[0] - ordered[1]

    agreement = 0.0
    if risk_numbers["dangerous_fire_index"] >= 0.7 and aggregate_relative_confidence["fire"] >= 0.55:
        agreement += 0.35
    if risk_numbers["fire_vs_controlled_gap"] > 0.15:
        agreement += 0.20
    if risk_numbers["spread_normalized"] > 0.35:
        agreement += 0.15
    if risk_numbers["flicker_normalized"] < 0.6:
        agreement += 0.10

    conflict = 0.0
    if abs(risk_numbers["fire_vs_controlled_gap"]) < 0.08:
        conflict += 0.2
    if risk_numbers["flicker_normalized"] > 0.75 and risk_numbers["spread_normalized"] < 0.2:
        conflict += 0.15

    base = 0.35 + 0.35 * dominance_gap + agreement - conflict

    if openai_output_optional and openai_output_optional.get("used"):
        base += 0.2 * float(openai_output_optional.get("confidence", 0.0))

    return max(0.0, min(1.0, base))


def assign_scenario_rank(
    final_score: float,
    previous_rank: str | None,
    thresholds: ScenarioThresholds,
    aggregate_relative_confidence: dict,
) -> str:
    prev = previous_rank
    fire = float(aggregate_relative_confidence.get("fire", 0.0))
    smoke = float(aggregate_relative_confidence.get("smoke", 0.0))

    if prev == "Emergency":
        if final_score >= thresholds.emergency_exit:
            return "Emergency"
    elif final_score >= thresholds.emergency_enter:
        return "Emergency"

    if prev == "Hazard":
        if final_score >= thresholds.hazard_exit:
            return "Hazard"
    elif final_score >= thresholds.hazard_enter:
        return "Hazard"

    if final_score <= thresholds.no_fire_risk_max:
        return "No Fire Risk"

    # Elevated Risk should mainly represent low-risk smoke-heavy scenes with little/no visible fire.
    if (
        final_score >= thresholds.elevated_enter
        and fire <= thresholds.elevated_max_fire_conf
        and smoke >= thresholds.elevated_min_smoke_conf
    ):
        return "Elevated Risk"

    # Otherwise, default to Hazard for non-trivial risk, especially when fire signal is present.
    return "Hazard"
