"""
Central configuration for F1 prediction system.
Update DRIVER_TEAMS each season as lineups change.
"""

# 2025 Season Driver-Team Mapping
DRIVER_TEAMS: dict[str, str] = {
    'VER': 'Red Bull',
    'LAW': 'Red Bull',
    'NOR': 'McLaren',
    'PIA': 'McLaren',
    'LEC': 'Ferrari',
    'HAM': 'Ferrari',
    'RUS': 'Mercedes',
    'ANT': 'Mercedes',
    'ALO': 'Aston Martin',
    'STR': 'Aston Martin',
    'GAS': 'Alpine',
    'DOO': 'Alpine',
    'TSU': 'Racing Bulls',
    'HAD': 'Racing Bulls',
    'ALB': 'Williams',
    'SAI': 'Williams',
    'HUL': 'Sauber',
    'BOR': 'Sauber',
    'OCO': 'Haas',
    'BEA': 'Haas',
}

# DNF probabilities per team (per lap, based on historical reliability)
DEFAULT_DNF_RATES: dict[str, float] = {
    'Red Bull': 0.0015,
    'McLaren': 0.0012,
    'Ferrari': 0.0018,
    'Mercedes': 0.0010,
    'Aston Martin': 0.0020,
    'Alpine': 0.0025,
    'Racing Bulls': 0.0022,
    'Williams': 0.0025,
    'Sauber': 0.0028,
    'Haas': 0.0025,
}

# Tire compound characteristics
TIRE_COMPOUNDS: dict[str, dict] = {
    'SOFT': {'pace_delta': -0.8, 'deg_rate': 0.08, 'optimal_laps': 15},
    'MEDIUM': {'pace_delta': 0.0, 'deg_rate': 0.05, 'optimal_laps': 25},
    'HARD': {'pace_delta': 0.6, 'deg_rate': 0.03, 'optimal_laps': 40},
    'INTERMEDIATE': {'pace_delta': 5.0, 'deg_rate': 0.02, 'optimal_laps': 30},
    'WET': {'pace_delta': 10.0, 'deg_rate': 0.01, 'optimal_laps': 50},
}

# Circuit-specific data (update per season)
CIRCUITS: dict[str, dict] = {
    'Bahrain': {'laps': 57, 'pit_loss': 21.0, 'drs_zones': 3, 'overtake_delta': 0.6},
    'Saudi Arabia': {'laps': 50, 'pit_loss': 20.0, 'drs_zones': 3, 'overtake_delta': 0.7},
    'Australia': {'laps': 58, 'pit_loss': 22.0, 'drs_zones': 4, 'overtake_delta': 0.5},
    'Japan': {'laps': 53, 'pit_loss': 23.0, 'drs_zones': 1, 'overtake_delta': 1.0},
    'China': {'laps': 56, 'pit_loss': 22.0, 'drs_zones': 2, 'overtake_delta': 0.6},
    'Miami': {'laps': 57, 'pit_loss': 21.0, 'drs_zones': 3, 'overtake_delta': 0.7},
    'Monaco': {'laps': 78, 'pit_loss': 24.0, 'drs_zones': 1, 'overtake_delta': 1.5},
    'Canada': {'laps': 70, 'pit_loss': 22.0, 'drs_zones': 2, 'overtake_delta': 0.6},
    'Spain': {'laps': 66, 'pit_loss': 21.0, 'drs_zones': 2, 'overtake_delta': 0.8},
    'Austria': {'laps': 71, 'pit_loss': 20.0, 'drs_zones': 3, 'overtake_delta': 0.5},
    'Great Britain': {'laps': 52, 'pit_loss': 21.0, 'drs_zones': 2, 'overtake_delta': 0.7},
    'Hungary': {'laps': 70, 'pit_loss': 22.0, 'drs_zones': 1, 'overtake_delta': 1.2},
    'Belgium': {'laps': 44, 'pit_loss': 23.0, 'drs_zones': 2, 'overtake_delta': 0.5},
    'Netherlands': {'laps': 72, 'pit_loss': 20.0, 'drs_zones': 2, 'overtake_delta': 1.0},
    'Italy': {'laps': 53, 'pit_loss': 26.0, 'drs_zones': 2, 'overtake_delta': 0.4},
    'Azerbaijan': {'laps': 51, 'pit_loss': 24.0, 'drs_zones': 2, 'overtake_delta': 0.5},
    'Singapore': {'laps': 62, 'pit_loss': 30.0, 'drs_zones': 3, 'overtake_delta': 1.1},
    'United States': {'laps': 56, 'pit_loss': 21.0, 'drs_zones': 2, 'overtake_delta': 0.7},
    'Mexico': {'laps': 71, 'pit_loss': 22.0, 'drs_zones': 3, 'overtake_delta': 0.6},
    'Brazil': {'laps': 71, 'pit_loss': 21.0, 'drs_zones': 2, 'overtake_delta': 0.5},
    'Las Vegas': {'laps': 50, 'pit_loss': 21.0, 'drs_zones': 2, 'overtake_delta': 0.6},
    'Qatar': {'laps': 57, 'pit_loss': 21.0, 'drs_zones': 2, 'overtake_delta': 0.8},
    'Abu Dhabi': {'laps': 58, 'pit_loss': 22.0, 'drs_zones': 2, 'overtake_delta': 0.7},
}

# Grid penalty types (positions)
PENALTY_TYPES: dict[str, int] = {
    'engine': 10,       # ICE, TC, MGU-H, MGU-K
    'full_pu': 20,      # Full power unit change (back of grid effectively)
    'gearbox': 5,       # Gearbox change
    'pitlane_start': 20,  # Starting from pit lane
}

# Track characteristics for similarity matching
# Categories: 'street' (Monaco, Singapore), 'high_speed' (Monza, Spa),
#             'technical' (Hungary, Spain), 'balanced' (Bahrain, Abu Dhabi)
TRACK_TYPES: dict[str, str] = {
    'Monaco': 'street',
    'Singapore': 'street',
    'Azerbaijan': 'street',
    'Las Vegas': 'street',
    'Saudi Arabia': 'street',
    'Italy': 'high_speed',
    'Belgium': 'high_speed',
    'Mexico': 'high_speed',
    'Qatar': 'high_speed',
    'Hungary': 'technical',
    'Spain': 'technical',
    'Netherlands': 'technical',
    'Japan': 'technical',
    'Bahrain': 'balanced',
    'Abu Dhabi': 'balanced',
    'Australia': 'balanced',
    'China': 'balanced',
    'Miami': 'balanced',
    'Canada': 'balanced',
    'Austria': 'balanced',
    'Great Britain': 'balanced',
    'United States': 'balanced',
    'Brazil': 'balanced',
}

def get_similar_tracks(track: str) -> list[str]:
    """Get tracks with similar characteristics."""
    track_type = TRACK_TYPES.get(track, 'balanced')
    return [t for t, tt in TRACK_TYPES.items() if tt == track_type and t != track]
