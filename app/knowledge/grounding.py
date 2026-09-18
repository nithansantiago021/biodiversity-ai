from sqlalchemy.orm import Session
from app.models.db_models import EnvironmentalObservation
from app.knowledge.retrieval import search_similar_chunks


def generate_observation_queries(observation: EnvironmentalObservation) -> list[str]:
    """
    Translates structured environmental metrics into targeted scientific search queries
    covering soil chemistry, land management, biodiversity, climate stress, and human impact.
    """
    queries = []

    # 1. Soil Organic Carbon (SOC)
    if observation.soil_organic_carbon < 1.0:
        queries.append(
            f"Impacts and risks of low soil organic carbon ({observation.soil_organic_carbon}%) in {observation.land_use} land use."
        )
    elif observation.soil_organic_carbon > 3.0:
        queries.append(
            f"Benefits of high soil organic carbon ({observation.soil_organic_carbon}%) for soil health and carbon sequestration."
        )

    # 2. Soil pH
    if observation.soil_ph < 5.5:
        queries.append(
            f"Effects of acidic soil pH ({observation.soil_ph}) on biodiversity and nutrient availability."
        )
    elif observation.soil_ph > 8.0:
        queries.append(
            f"Effects of alkaline soil pH ({observation.soil_ph}) on soil biological activity."
        )

    # 3. Soil Moisture
    if observation.soil_moisture < 15.0:
        queries.append(
            f"Impact of low soil moisture ({observation.soil_moisture}%) and drought stress on soil organisms."
        )

    # 4. Biodiversity Metrics (Species Richness & Habitat Diversity)
    if observation.species_richness < 10.0 or observation.habitat_diversity < 0.3:
        queries.append(
            f"Drivers of low species richness ({observation.species_richness}) and low habitat diversity ({observation.habitat_diversity}) in {observation.land_cover}."
        )

    # 5. Climate Stress (Temperature & Rainfall)
    if observation.temperature > 32.0 and observation.rainfall < 500.0:
        queries.append(
            f"Combined effect of high temperature ({observation.temperature}°C) and low rainfall ({observation.rainfall}mm) on soil ecosystem functions."
        )
    elif observation.rainfall < 400.0:
        queries.append(
            f"Effects of low annual precipitation ({observation.rainfall}mm) on soil biodiversity and microbial activity."
        )

    # 6. Human Impact (Pollution Index & Deforestation Rate)
    if observation.pollution_index > 0.3:
        queries.append(
            f"Ecotoxicological effects of soil pollution index ({observation.pollution_index}) on soil fauna and biological functioning."
        )

    if observation.deforestation_rate > 0.05:
        queries.append(
            f"Consequences of high deforestation rate ({observation.deforestation_rate}) on species richness and habitat fragmentation."
        )

    # 7. Fallback Query
    if not queries:
        queries.append(
            f"Soil biodiversity and ecosystem function in {observation.land_use} with {observation.land_cover} cover."
        )

    return queries


def analyze_observation_with_grounding(
    db: Session,
    observation: EnvironmentalObservation,
    top_k_per_query: int = 2,
) -> dict:
    """
    Generates targeted queries from an EnvironmentalObservation across all metrics,
    retrieves supporting scientific chunks with provenance, and returns a grounded payload.
    """
    generated_queries = generate_observation_queries(observation)
    evidence_items = []

    for query in generated_queries:
        retrieved_chunks = search_similar_chunks(
            db, query_text=query, top_k=top_k_per_query
        )
        for item in retrieved_chunks:
            evidence_items.append(
                {
                    "query_trigger": query,
                    "chunk_id": item["chunk_id"],
                    "page_number": item["page_number"],
                    "rerank_score": item["rerank_score"],
                    "chunk_text": item["chunk_text"],
                    "provenance": item["provenance"],
                }
            )

    return {
        "observation_id": observation.id,
        "location": {
            "latitude": observation.latitude,
            "longitude": observation.longitude,
        },
        "metrics": {
            "soil_ph": observation.soil_ph,
            "soil_organic_carbon": observation.soil_organic_carbon,
            "soil_moisture": observation.soil_moisture,
            "land_use": observation.land_use,
            "land_cover": observation.land_cover,
            "species_richness": observation.species_richness,
            "habitat_diversity": observation.habitat_diversity,
            "temperature": observation.temperature,
            "rainfall": observation.rainfall,
            "pollution_index": observation.pollution_index,
            "deforestation_rate": observation.deforestation_rate,
        },
        "generated_queries": generated_queries,
        "scientific_evidence": evidence_items,
    }
