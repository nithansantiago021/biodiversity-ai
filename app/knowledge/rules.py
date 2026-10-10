from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.models.db_models import EnvironmentalObservation


@dataclass
class RuleResult:
    rule_id: str
    directive: str  # instruction injected into the LLM prompt
    blocks_keywords: List[str] = field(default_factory=list)
    reason: str = ""  # human-readable reason, surfaced in the response trace


def evaluate_rules(obs: EnvironmentalObservation) -> List[RuleResult]:
    """Evaluates every rule against a persisted observation and returns the ones that fired."""
    results: List[RuleResult] = []

    # R1 -- pollinator collapse under a fixed pesticide schedule.
    if (
        obs.pesticide_use is True
        and obs.species_richness is not None
        and obs.species_richness < 15
    ):
        results.append(
            RuleResult(
                rule_id="R1_pollinator_spray_conflict",
                directive=(
                    "The site sprays pesticides on a fixed schedule and species richness is low. "
                    "Do NOT recommend flower strips, pollinator nesting habitat, or other interventions "
                    "that concentrate beneficial insects near the sprayed area -- they become ecological "
                    "traps. Recommend integrated pest management (IPM) or a revised spray regime FIRST, "
                    "ahead of any pollinator-habitat intervention."
                ),
                blocks_keywords=[
                    "flower strip",
                    "pollinator habitat",
                    "nesting habitat",
                    "wildflower",
                ],
                reason=(
                    "Fixed pesticide schedule + low species richness: pollinator-habitat interventions "
                    "become ecological traps unless the spray regime changes first."
                ),
            )
        )

    # R2 -- low carbon + low rainfall + monoculture: biomass-heavy interventions
    # compete with the crop for scarce water.
    if (
        obs.soil_organic_carbon is not None
        and obs.soil_organic_carbon < 1.0
        and obs.rainfall is not None
        and obs.rainfall < 450
        and obs.land_use is not None
        and "mono" in obs.land_use.lower()
    ):
        results.append(
            RuleResult(
                rule_id="R2_carbon_water_competition",
                directive=(
                    "Soil organic carbon is low, rainfall is low, and land use is monoculture. "
                    "Agroforestry and other high-transpiration biomass interventions must be sequenced "
                    "AFTER water-conservation measures (mulching, water harvesting, reduced tillage) -- "
                    "added biomass competes with the crop for scarce water at this rainfall level."
                ),
                blocks_keywords=["agroforestry", "tree planting", "afforestation"],
                reason=(
                    "Low SOC + low rainfall + monoculture: added transpiring biomass competes with the "
                    "crop for water unless conservation measures come first."
                ),
            )
        )

    # R3 -- uncontrolled grazing: establishment-phase interventions fail
    # before they take root.
    if (
        obs.grazing_controlled is False
        and obs.habitat_diversity is not None
        and obs.habitat_diversity < 0.3
    ):
        results.append(
            RuleResult(
                rule_id="R3_uncontrolled_grazing",
                directive=(
                    "Grazing on this land is uncontrolled and habitat diversity is very low. Any "
                    "establishment-phase intervention (reseeding, tree planting, restoration plots) will "
                    "fail without a grazing management plan first. State this precondition explicitly and "
                    "recommend a grazing/institutional intervention as the first step."
                ),
                blocks_keywords=["reseeding", "restoration plot"],
                reason=(
                    "Uncontrolled grazing: establishment-phase interventions have no chance without a "
                    "grazing management precondition."
                ),
            )
        )

    # R4 -- acidic soil + high pollution: compounding chemical stress.
    if (
        obs.soil_ph is not None
        and obs.soil_ph < 5.5
        and obs.pollution_index is not None
        and obs.pollution_index > 0.3
    ):
        results.append(
            RuleResult(
                rule_id="R4_acidity_pollution_coupling",
                directive=(
                    "Soil is acidic AND the pollution index is elevated -- acidity increases the "
                    "bioavailability of many pollutants, so these compound rather than act independently. "
                    "Recommend a soil amendment (e.g. liming) before or alongside biological interventions, "
                    "and state this coupling explicitly in the ecological summary."
                ),
                blocks_keywords=[],
                reason=(
                    "Acidic soil increases pollutant bioavailability: biological interventions alone "
                    "under-perform without addressing acidity first."
                ),
            )
        )

    return results


def build_constraints_block(rule_results: List[RuleResult]) -> str:
    """Renders fired rules as a prompt section. Empty string if nothing fired."""
    if not rule_results:
        return ""
    lines = [f"- {r.directive}" for r in rule_results]
    return "MANDATORY CONSTRAINTS (deterministic, not optional):\n" + "\n".join(lines)


def apply_post_filters(
    rule_results: List[RuleResult], recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Deterministically demotes any LLM-produced recommendation matching a fired
    rule's blocked keywords, REGARDLESS of whether the LLM followed the
    prompt's directive. Unplug the LLM's compliance and this still fires.
    """
    blocked_terms = [kw for r in rule_results for kw in r.blocks_keywords]
    if not blocked_terms:
        return recommendations

    for rec in recommendations:
        haystack = f"{rec.get('title', '')} {rec.get('description', '')}".lower()
        for term in blocked_terms:
            if term in haystack:
                rec["confidence"] = "low"
                rec["time_horizon"] = (
                    f"long-term (deferred: constrained by '{term}' rule)"
                )
                existing_tradeoff = rec.get("tradeoffs", "") or ""
                rec["tradeoffs"] = (
                    existing_tradeoff
                    + f" [Automated constraint: '{term}' demoted by deterministic rule check.]"
                ).strip()
    return recommendations
