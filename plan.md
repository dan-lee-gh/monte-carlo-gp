# Monte Carlo GP - F1 Race Prediction System

## Architecture Overview

```text
+-----------------+     +------------------+     +-----------------+
|    Data Layer   | --> |   Model Layer    | --> |   Simulation    |
| FastF1 / APIs   |     | Quali & Race ML  |     | Monte Carlo GP  |
| Live / Manual   |     | Ratings & Priors |     | Race Engine     |
+-----------------+     +------------------+     +-----------------+
                                |
                                v
                         +--------------+
                         |   Outputs    |
                         | Pole / Win   |
                         | Podium dists |
                         +--------------+
```

## 1. Data Layer

### Primary Data Sources

**FastF1 (Python Library)** - telemetry, laps, sectors, tires, weather.

```python
import fastf1

session = fastf1.get_session(2025, 'Abu Dhabi', 'FP2')
session.load()

laps = session.laps
norris_laps = laps.pick_driver('NOR')
fastest_lap = norris_laps.pick_fastest()
telemetry = fastest_lap.get_telemetry() if fastest_lap is not None else None
sector_times = laps[['Driver', 'Sector1Time', 'Sector2Time', 'Sector3Time']]
```

**Jolpica-F1 API** - historical results/standings (Ergast replacement, same schema).

```python
import requests

# Jolpica-F1 is the community continuation of Ergast (deprecated end of 2024)
url = "https://api.jolpi.ca/ergast/f1/2024/results.json"
data = requests.get(url).json()
```

**OpenF1 API** - real-time session data.

```python
url = "https://api.openf1.org/v1/car_data?session_key=latest"
```

### Data Schema

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class LapData:
    session_id: str
    driver: str
    team: str
    lap_number: int
    lap_time: float  # seconds
    sector_1: float
    sector_2: float
    sector_3: float
    compound: str  # SOFT, MEDIUM, HARD
    tire_life: int  # laps on this set
    fuel_load: Optional[float]
    is_valid: bool
    track_status: str  # GREEN, YELLOW, SC, VSC

@dataclass
class QualifyingResult:
    season: int
    round: int
    driver: str
    team: str
    q1_time: Optional[float]
    q2_time: Optional[float]
    q3_time: Optional[float]
    grid_position: int

@dataclass
class RaceResult:
    season: int
    round: int
    driver: str
    team: str
    grid: int
    finish: int
    status: str  # FINISHED, DNF, DSQ
    points: float
    fastest_lap: bool
```

### Storage

For a personal project, use SQLite or DuckDB (fast analytics).

```python
import duckdb

con = duckdb.connect('f1_data.db')

con.execute(
    """
    CREATE TABLE IF NOT EXISTS laps (
        session_id VARCHAR,
        driver VARCHAR,
        team VARCHAR,
        lap_number INTEGER,
        lap_time DOUBLE,
        sector_1 DOUBLE,
        sector_2 DOUBLE,
        sector_3 DOUBLE,
        compound VARCHAR,
        tire_life INTEGER,
        PRIMARY KEY (session_id, driver, lap_number)
    )
    """
)
```

For production scale, use PostgreSQL + TimescaleDB for telemetry.

## 2. Data Loading

```python
import fastf1
from pathlib import Path
import pandas as pd

class F1DataLoader:
    def __init__(self, cache_dir: str = './cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))
        self._session_cache: dict[tuple, pd.DataFrame] = {}

    def load_session(self, season: int, race: str, session: str) -> pd.DataFrame:
        """Load session data with caching. Session: FP1, FP2, FP3, Q, SQ, S, R"""
        cache_key = (season, race, session)

        # Return cached successful results only
        if cache_key in self._session_cache:
            return self._session_cache[cache_key]

        try:
            sess = fastf1.get_session(season, race, session)
            sess.load()
            laps = sess.laps
            # Only cache non-empty results
            if not laps.empty:
                self._session_cache[cache_key] = laps
            return laps
        except Exception as e:
            print(f"Warning: Could not load {season} {race} {session}: {e}")
            return pd.DataFrame()

    def load_season_data(self, season: int) -> dict:
        """Load all qualifying and race results for a season."""
        from datetime import datetime, timezone

        schedule = fastf1.get_event_schedule(season)
        results = {'qualifying': [], 'races': [], 'sprints': [], 'sprint_qualifying': []}
        today = datetime.now(timezone.utc)

        for _, event in schedule.iterrows():
            if event['EventFormat'] == 'testing':
                continue
            # Skip future events (no data available yet)
            event_date = pd.to_datetime(event.get('EventDate', event.get('Session5Date')))
            if pd.notna(event_date):
                # Make comparison timezone-aware
                if event_date.tzinfo is None:
                    event_date = event_date.tz_localize('UTC')
                if event_date > today:
                    continue
            try:
                q_session = self.load_session(season, event['EventName'], 'Q')
                if not q_session.empty:
                    results['qualifying'].append(self._extract_quali_results(q_session))

                r_session = self.load_session(season, event['EventName'], 'R')
                if not r_session.empty:
                    results['races'].append(self._extract_race_results(r_session))

                # Handle sprint weekends (format names: 'sprint', 'sprint_shootout', 'sprint_qualifying')
                event_format = str(event.get('EventFormat', '')).lower()
                if 'sprint' in event_format:
                    # Sprint Shootout (SQ) - qualifying for sprint race
                    sq_session = self.load_session(season, event['EventName'], 'SQ')
                    if not sq_session.empty:
                        results['sprint_qualifying'].append(self._extract_quali_results(sq_session))

                    # Sprint race (S)
                    s_session = self.load_session(season, event['EventName'], 'S')
                    if not s_session.empty:
                        results['sprints'].append(self._extract_race_results(s_session))
            except Exception as e:
                print(f"Warning: Could not load {event['EventName']}: {e}")

        return results

    def _extract_quali_results(self, laps: pd.DataFrame) -> list[tuple[str, float]]:
        """Extract (driver, best_time) sorted by qualifying position."""
        # Filter to valid laps only (exclude pit laps, track limit violations, etc.)
        # FastF1 marks accurate laps with IsAccurate=True
        if 'IsAccurate' in laps.columns:
            # fillna(False) treats NaN as not accurate (conservative)
            valid_laps = laps[laps['IsAccurate'].fillna(False) == True]
        else:
            # Fallback: filter out obvious invalid laps (pit in/out, deleted times)
            mask = pd.Series(True, index=laps.index)
            if 'PitInTime' in laps.columns:
                mask = mask & (laps['PitInTime'].isna())
            if 'PitOutTime' in laps.columns:
                mask = mask & (laps['PitOutTime'].isna())
            if 'Deleted' in laps.columns:
                # fillna(False) treats NaN as not deleted (lap is valid)
                mask = mask & (laps['Deleted'].fillna(False) != True)
            valid_laps = laps[mask]

        if valid_laps.empty:
            valid_laps = laps  # Fallback to all laps if no valid ones found

        best_times = valid_laps.groupby('Driver')['LapTime'].min().sort_values()
        # Filter out NaT values (drivers with no valid lap times)
        return [
            (d, t.total_seconds())
            for d, t in best_times.items()
            if pd.notna(t)
        ]

    def _extract_race_results(self, laps: pd.DataFrame) -> list[tuple[str, int]]:
        """Extract (driver, finish_position)."""
        if laps.empty or 'Position' not in laps.columns:
            return []
        # Sort by lap number to ensure .last() gets the actual final lap
        if 'LapNumber' in laps.columns:
            laps = laps.sort_values('LapNumber')
        final_laps = laps.groupby('Driver').last()
        results = []
        for d, row in final_laps.iterrows():
            pos = row.get('Position')
            # Skip DNF/DNS drivers with NaN position
            if pd.notna(pos):
                results.append((d, int(pos)))
        return results

    def get_weather(self, season: int, race: str, session: str) -> dict:
        """Get weather data for a session."""
        default = {'air_temp': 25, 'track_temp': 35, 'humidity': 50, 'rainfall': False, 'wind_speed': 5}
        try:
            sess = fastf1.get_session(season, race, session)
            sess.load(weather=True, laps=False, telemetry=False, messages=False)
            weather = sess.weather_data
            if weather is not None and not weather.empty:
                # Helper to safely get mean with NaN handling
                def safe_mean(col: str, fallback: float) -> float:
                    if col not in weather.columns:
                        return fallback
                    val = weather[col].mean()
                    return float(val) if pd.notna(val) else fallback

                return {
                    'air_temp': safe_mean('AirTemp', default['air_temp']),
                    'track_temp': safe_mean('TrackTemp', default['track_temp']),
                    'humidity': safe_mean('Humidity', default['humidity']),
                    'rainfall': bool(weather['Rainfall'].any()) if 'Rainfall' in weather.columns else False,
                    'wind_speed': safe_mean('WindSpeed', default['wind_speed']),
                }
        except Exception:
            pass
        return default
```

## 3. Feature Engineering

Domain-heavy step for quali/race predictors.

```python
import fastf1
import pandas as pd
import numpy as np

class F1FeatureEngine:
    def __init__(self, data_loader: F1DataLoader):
        self.loader = data_loader
        self.laps: pd.DataFrame = pd.DataFrame()
        self.race_results: pd.DataFrame = pd.DataFrame()  # Grid/finish positions for overtake analysis

    def load_historical_data(self, seasons: list[int]):
        """Load and combine lap data from multiple seasons."""
        from datetime import datetime, timezone

        all_laps = []
        all_results = []
        today = datetime.now(timezone.utc)

        for season in seasons:
            schedule = fastf1.get_event_schedule(season)
            for _, event in schedule.iterrows():
                if event['EventFormat'] == 'testing':
                    continue
                # Skip future events (no data available yet)
                event_date = pd.to_datetime(event.get('EventDate', event.get('Session5Date')))
                if pd.notna(event_date):
                    # Make comparison timezone-aware
                    if event_date.tzinfo is None:
                        event_date = event_date.tz_localize('UTC')
                    if event_date > today:
                        continue
                for session_type in ['FP1', 'FP2', 'FP3', 'Q', 'SQ', 'S', 'R']:  # SQ = Sprint Shootout
                    laps = self.loader.load_session(season, event['EventName'], session_type)
                    if not laps.empty:
                        laps = laps.copy()
                        laps['season'] = season
                        laps['circuit'] = event['EventName']
                        laps['session_type'] = session_type
                        all_laps.append(laps)

                # Load race results for grid/finish analysis
                try:
                    race_sess = fastf1.get_session(season, event['EventName'], 'R')
                    race_sess.load()
                    if hasattr(race_sess, 'results') and not race_sess.results.empty:
                        results = race_sess.results[['Abbreviation', 'GridPosition', 'Position', 'Status']].copy()
                        results.columns = ['driver', 'grid', 'finish', 'status']
                        results['season'] = season
                        results['circuit'] = event['EventName']
                        all_results.append(results)
                except Exception:
                    pass  # Skip if results unavailable

        self.laps = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
        self.race_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    def calculate_quali_features(self, driver: str, circuit: str) -> dict:
        """Generate features for qualifying prediction."""
        # Get last ~100 qualifying laps (5 recent races × ~20 laps per quali session)
        RECENT_LAPS = 5 * 20
        recent = self.laps[
            (self.laps['Driver'] == driver) &
            (self.laps['session_type'] == 'Q')
        ].tail(RECENT_LAPS)

        return {
            'avg_gap_to_pole_pct': self._calc_gap_to_pole(recent),
            'teammate_delta': self._calc_teammate_delta(driver, recent),
            'circuit_affinity': self._calc_circuit_affinity(driver, circuit),
            'q3_conversion': self._calc_q3_rate(driver),
            'low_speed_strength': self._calc_sector_strength(driver, 'low'),
            'high_speed_strength': self._calc_sector_strength(driver, 'high'),
            'traction_strength': self._calc_sector_strength(driver, 'traction'),
            'form_score': self._calc_form_score(driver, decay=0.85),
        }

    def calculate_race_features(self, driver: str, circuit: str, weather: dict) -> dict:
        """Generate features for race prediction including weather."""
        # Validate all weather values with safe defaults
        rainfall = weather.get('rainfall', False)
        # Handle various rainfall representations: bool, numeric, or string
        if pd.isna(rainfall):
            is_wet = False
        elif isinstance(rainfall, bool):
            is_wet = rainfall
        elif isinstance(rainfall, (int, float)):
            is_wet = rainfall > 0
        else:
            is_wet = str(rainfall).lower() in ('true', 'yes', '1')
        track_temp = weather.get('track_temp', 35)
        track_temp = float(track_temp) if pd.notna(track_temp) else 35.0

        return {
            'race_pace_delta': self._calc_race_pace(driver),
            'deg_rate': self._calc_deg_rate(driver),
            'overtake_rate': self._calc_overtake_rate(driver),
            'position_hold_rate': self._calc_defense_rate(driver),
            'dnf_probability': self._calc_dnf_prob(driver),
            'start_delta': self._calc_start_performance(driver),
            'clutch_factor': self._calc_clutch_factor(driver),
            # Weather-adjusted features
            'wet_performance': self._calc_wet_performance(driver) if is_wet else 0,
            'high_temp_deg': self._calc_temp_sensitivity(driver) * track_temp,
        }

    def _calc_gap_to_pole(self, laps: pd.DataFrame) -> float:
        """Average percentage gap to pole position in recent qualis."""
        if laps.empty:
            return 1.5  # default 1.5% gap
        # Implementation: calculate gap to session-best lap time
        return 1.0

    def _calc_teammate_delta(self, driver: str, laps: pd.DataFrame) -> float:
        """Average lap time delta vs teammate.

        Positive = faster than teammate, negative = slower.
        Returns percentage delta (e.g., 0.1 = 0.1% faster).
        """
        if laps.empty or 'Team' not in laps.columns:
            return 0.0

        driver_laps = laps[laps['Driver'] == driver]
        if driver_laps.empty:
            return 0.0

        # Get driver's team
        team = driver_laps['Team'].iloc[0]
        team_laps = laps[laps['Team'] == team]

        # Find teammate(s)
        teammates = [d for d in team_laps['Driver'].unique() if d != driver]
        if not teammates:
            return 0.0

        deltas = []
        # Compare session by session
        for (season, circuit), session_laps in laps.groupby(['season', 'circuit']):
            driver_session = session_laps[session_laps['Driver'] == driver]
            if driver_session.empty:
                continue

            driver_best = driver_session['LapTime'].min()
            if pd.isna(driver_best):
                continue
            driver_sec = driver_best.total_seconds() if hasattr(driver_best, 'total_seconds') else float(driver_best)

            for teammate in teammates:
                tm_session = session_laps[session_laps['Driver'] == teammate]
                if tm_session.empty:
                    continue
                tm_best = tm_session['LapTime'].min()
                if pd.isna(tm_best):
                    continue
                tm_sec = tm_best.total_seconds() if hasattr(tm_best, 'total_seconds') else float(tm_best)

                if tm_sec > 0:
                    # Positive = driver faster than teammate
                    delta_pct = (tm_sec - driver_sec) / tm_sec * 100
                    deltas.append(delta_pct)

        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    def _calc_circuit_affinity(self, driver: str, circuit: str) -> float:
        """Historical performance at this circuit vs average.

        Returns positive value if driver performs better at this circuit,
        negative if worse. Range typically -0.5 to 0.5.
        """
        if self.laps.empty:
            return 0.0

        driver_laps = self.laps[self.laps['Driver'] == driver]
        if driver_laps.empty:
            return 0.0

        # Get driver's average gap to fastest at this circuit
        circuit_laps = driver_laps[driver_laps['circuit'] == circuit]
        if circuit_laps.empty or 'LapTime' not in circuit_laps.columns:
            return 0.0

        # Calculate gap to session fastest for circuit vs overall
        def calc_avg_gap(laps_subset):
            if laps_subset.empty:
                return None
            valid = laps_subset.dropna(subset=['LapTime'])
            if valid.empty:
                return None
            times = valid['LapTime'].apply(
                lambda x: x.total_seconds() if hasattr(x, 'total_seconds') else float(x)
            )
            return times.mean() if len(times) > 0 else None

        circuit_avg = calc_avg_gap(circuit_laps)
        overall_avg = calc_avg_gap(driver_laps)

        if circuit_avg is None or overall_avg is None or overall_avg == 0:
            return 0.0

        # Positive = faster at this circuit than average
        # Normalized to roughly -0.5 to 0.5 range
        affinity = (overall_avg - circuit_avg) / overall_avg
        return max(-0.5, min(0.5, affinity))

    def _calc_q3_rate(self, driver: str) -> float:
        """Rate of Q3 appearances based on historical data."""
        if self.laps.empty:
            return 0.5  # Default 50% for unknown drivers

        # Q3 participants are top 10 in qualifying
        q_laps = self.laps[
            (self.laps['Driver'] == driver) &
            (self.laps['session_type'] == 'Q')
        ]
        if q_laps.empty:
            return 0.5

        # Count sessions where driver had Q3 laps (made it to Q3)
        # Q3 is typically identified by having laps after the Q2 cutoff
        sessions = q_laps.groupby(['season', 'circuit']).size()
        total_sessions = len(sessions)
        if total_sessions == 0:
            return 0.5

        # Approximate: if driver has many laps in a session, likely made Q3
        # More accurate would be to check actual qualifying results
        q3_count = sum(1 for count in sessions if count >= 3)  # At least 3 laps suggests Q3
        return q3_count / total_sessions

    def _calc_sector_strength(self, driver: str, sector_type: str) -> float:
        """Performance in specific sector types relative to field.

        Args:
            sector_type: 'high_speed', 'low_speed', or 'traction'

        Returns:
            Percentage advantage/disadvantage in sector type (e.g., 0.1 = 0.1% faster)
        """
        if self.laps.empty:
            return 0.0

        # Sector columns in FastF1 data
        sector_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
        if not all(col in self.laps.columns for col in sector_cols):
            return 0.0

        driver_laps = self.laps[self.laps['Driver'] == driver]
        if driver_laps.empty:
            return 0.0

        # Map circuit sectors to types (simplified - would need circuit-specific mapping)
        # For now, use heuristic: S1 often high-speed, S2 mixed, S3 often traction
        sector_type_map = {
            'high_speed': 'Sector1Time',
            'low_speed': 'Sector2Time',
            'traction': 'Sector3Time',
        }
        target_sector = sector_type_map.get(sector_type, 'Sector1Time')

        deltas = []
        for (season, circuit), session_laps in self.laps.groupby(['season', 'circuit']):
            driver_session = session_laps[session_laps['Driver'] == driver]
            if driver_session.empty or target_sector not in driver_session.columns:
                continue

            # Get driver's best sector time
            driver_sectors = driver_session[target_sector].dropna()
            if driver_sectors.empty:
                continue
            driver_best = driver_sectors.min()
            driver_sec = driver_best.total_seconds() if hasattr(driver_best, 'total_seconds') else float(driver_best)

            # Get session best for this sector
            session_sectors = session_laps[target_sector].dropna()
            if session_sectors.empty:
                continue
            session_best = session_sectors.min()
            session_sec = session_best.total_seconds() if hasattr(session_best, 'total_seconds') else float(session_best)

            if session_sec > 0:
                # Positive = driver faster than field average in this sector
                delta_pct = (session_sec - driver_sec) / session_sec * 100
                deltas.append(delta_pct)

        if not deltas:
            return 0.0
        # Clamp to reasonable range
        result = sum(deltas) / len(deltas)
        return max(-1.0, min(1.0, result))

    def _calc_form_score(self, driver: str, decay: float = 0.85) -> float:
        """Recent form with exponential decay weighting.

        Returns a score between -1 (poor form) and 1 (excellent form).
        Based on recent qualifying positions weighted by recency.
        """
        if self.laps.empty:
            return 0.0

        # Get recent qualifying results for driver
        q_laps = self.laps[
            (self.laps['Driver'] == driver) &
            (self.laps['session_type'] == 'Q')
        ]
        if q_laps.empty:
            return 0.0

        # Group by session to get finishing positions
        sessions = q_laps.groupby(['season', 'circuit']).agg({
            'LapTime': 'min'
        }).reset_index()

        if len(sessions) == 0:
            return 0.0

        # Calculate relative performance for each session and apply decay weighting
        # Compare driver's best time to session's overall best to measure performance
        scores = []
        n_sessions = len(sessions)

        # Get session best times for comparison
        all_q_laps = self.laps[self.laps['session_type'] == 'Q']
        if all_q_laps.empty:
            return 0.0

        for idx, row in sessions.iterrows():
            season, circuit = row['season'], row['circuit']
            driver_best = row['LapTime']

            # Get overall session best for this race
            session_laps = all_q_laps[
                (all_q_laps['season'] == season) & (all_q_laps['circuit'] == circuit)
            ]
            if session_laps.empty or pd.isna(driver_best):
                continue

            session_best = session_laps['LapTime'].min()
            if pd.isna(session_best):
                continue

            # Convert to seconds for comparison
            driver_sec = driver_best.total_seconds() if hasattr(driver_best, 'total_seconds') else float(driver_best)
            session_sec = session_best.total_seconds() if hasattr(session_best, 'total_seconds') else float(session_best)

            if session_sec <= 0:
                continue

            # Calculate gap percentage (negative = faster than average, positive = slower)
            gap_pct = (driver_sec - session_sec) / session_sec * 100

            # Convert to score: 0% gap = 1.0, 1% gap = 0.0, 2% gap = -1.0
            performance_score = max(-1.0, min(1.0, 1.0 - gap_pct))

            # Apply decay weight (most recent = highest weight)
            weight = decay ** (n_sessions - 1 - idx)
            scores.append(performance_score * weight)

        if not scores:
            return 0.0

        # Weighted average of performance scores
        total_weight = sum(decay ** (n_sessions - 1 - i) for i in range(len(scores)))
        return sum(scores) / total_weight if total_weight > 0 else 0.0

    def _calc_race_pace(self, driver: str) -> float:
        """Average race pace delta to leader in race sessions.

        Returns percentage gap to race leader pace (e.g., 0.5 = 0.5% slower).
        Negative = faster than average, positive = slower.
        """
        if self.laps.empty:
            return 0.0

        # Filter for race laps only
        race_laps = self.laps[self.laps['session_type'] == 'R']
        if race_laps.empty:
            return 0.0

        driver_laps = race_laps[race_laps['Driver'] == driver]
        if driver_laps.empty:
            return 0.0

        deltas = []
        for (season, circuit), session_laps in race_laps.groupby(['season', 'circuit']):
            driver_session = session_laps[session_laps['Driver'] == driver]
            if driver_session.empty:
                continue

            # Get median lap time (excludes outliers from pit stops, SC)
            driver_times = driver_session['LapTime'].dropna()
            if len(driver_times) < 5:  # Need enough laps for meaningful comparison
                continue
            driver_median = driver_times.median()
            driver_sec = driver_median.total_seconds() if hasattr(driver_median, 'total_seconds') else float(driver_median)

            # Get leader's median pace
            leader_times = []
            for d in session_laps['Driver'].unique():
                d_times = session_laps[session_laps['Driver'] == d]['LapTime'].dropna()
                if len(d_times) >= 5:
                    med = d_times.median()
                    sec = med.total_seconds() if hasattr(med, 'total_seconds') else float(med)
                    leader_times.append(sec)

            if not leader_times:
                continue
            leader_pace = min(leader_times)

            if leader_pace > 0:
                delta_pct = (driver_sec - leader_pace) / leader_pace * 100
                deltas.append(delta_pct)

        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    def _calc_deg_rate(self, driver: str) -> float:
        """Tire degradation rate (seconds per lap) from stint analysis.

        Analyzes lap time progression during stints to estimate degradation.
        """
        if self.laps.empty:
            return 0.05  # Default

        # Filter for race laps (where deg matters most)
        race_laps = self.laps[self.laps['session_type'] == 'R']
        driver_laps = race_laps[race_laps['Driver'] == driver]
        if driver_laps.empty:
            return 0.05

        deg_rates = []
        for (season, circuit), session_laps in driver_laps.groupby(['season', 'circuit']):
            times = session_laps.sort_values('LapNumber' if 'LapNumber' in session_laps.columns else 'lap_number')
            lap_times = times['LapTime'].dropna()
            if len(lap_times) < 10:
                continue

            # Convert to seconds
            secs = [t.total_seconds() if hasattr(t, 'total_seconds') else float(t) for t in lap_times]

            # Simple linear regression to find degradation slope
            # Filter out obvious outliers (pit laps, SC)
            median_time = np.median(secs)
            filtered = [(i, t) for i, t in enumerate(secs) if abs(t - median_time) < 10]
            if len(filtered) < 5:
                continue

            x = np.array([f[0] for f in filtered])
            y = np.array([f[1] for f in filtered])

            # Linear fit: y = mx + b, m is degradation per lap
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                if 0 < slope < 0.5:  # Reasonable deg range
                    deg_rates.append(slope)

        if not deg_rates:
            return 0.05
        return max(0.01, min(0.15, sum(deg_rates) / len(deg_rates)))

    def _calc_overtake_rate(self, driver: str) -> float:
        """Positions gained per race on average from grid to finish.

        Positive = gains positions, negative = loses positions.
        """
        if self.race_results.empty if hasattr(self, 'race_results') else True:
            return 0.0

        driver_races = self.race_results[self.race_results['driver'] == driver]
        if driver_races.empty:
            return 0.0

        gains = []
        for _, race in driver_races.iterrows():
            grid = race.get('grid', 0)
            finish = race.get('finish', 0)
            if grid > 0 and finish > 0:
                # Positive = gained positions (grid was worse than finish)
                gains.append(grid - finish)

        if not gains:
            return 0.0
        return sum(gains) / len(gains)

    def _calc_defense_rate(self, driver: str) -> float:
        """Success rate at defending position based on historical battles.

        Returns probability of successfully defending (0.0 to 1.0).
        """
        # Without detailed position tracking data, estimate from results
        if self.race_results.empty if hasattr(self, 'race_results') else True:
            return 0.7  # Default - most drivers hold position

        driver_races = self.race_results[self.race_results['driver'] == driver]
        if len(driver_races) < 3:
            return 0.7

        # Estimate: if driver consistently finishes near or ahead of grid, good defender
        hold_count = 0
        total = 0
        for _, race in driver_races.iterrows():
            grid = race.get('grid', 0)
            finish = race.get('finish', 0)
            if grid > 0 and finish > 0:
                total += 1
                # Defended if finished at or ahead of grid position
                if finish <= grid + 1:  # Allow 1 position loss
                    hold_count += 1

        if total == 0:
            return 0.7
        return max(0.3, min(0.95, hold_count / total))

    def _calc_dnf_prob(self, driver: str) -> float:
        """Historical DNF rate (per race probability).

        Considers both driver-specific and team reliability.
        """
        if self.race_results.empty if hasattr(self, 'race_results') else True:
            return 0.05  # Default 5%

        driver_races = self.race_results[self.race_results['driver'] == driver]
        if len(driver_races) < 5:
            return 0.05

        dnf_count = 0
        for _, race in driver_races.iterrows():
            status = str(race.get('status', 'Finished')).lower()
            if 'finished' not in status and '+' not in status:
                # DNF statuses include: collision, mechanical, accident, etc.
                dnf_count += 1

        dnf_rate = dnf_count / len(driver_races)
        # Clamp to reasonable range
        return max(0.01, min(0.20, dnf_rate))

    def _calc_start_performance(self, driver: str) -> float:
        """Average positions gained/lost at race start (lap 1).

        Positive = tends to gain positions at start.
        """
        if self.race_results.empty if hasattr(self, 'race_results') else True:
            return 0.0

        # Estimate from grid vs finish, weighted toward early finishers
        driver_races = self.race_results[self.race_results['driver'] == driver]
        if len(driver_races) < 5:
            return 0.0

        # Known good starters gain positions early; approximate with grid-finish delta
        # filtered by races where they finished near the front
        start_gains = []
        for _, race in driver_races.iterrows():
            grid = race.get('grid', 0)
            finish = race.get('finish', 0)
            if grid > 0 and finish > 0 and finish <= 15:  # Finished in points
                gain = grid - finish
                start_gains.append(gain * 0.4)  # Assume 40% of gains happen at start

        if not start_gains:
            return 0.0
        avg = sum(start_gains) / len(start_gains)
        return max(-3.0, min(3.0, avg))

    def _calc_clutch_factor(self, driver: str) -> float:
        """Performance in high-pressure situations (close finishes, title fights).

        Returns modifier: positive = performs better under pressure.
        Range typically -0.5 to 0.5.
        """
        if self.race_results.empty if hasattr(self, 'race_results') else True:
            return 0.0

        driver_races = self.race_results[self.race_results['driver'] == driver]
        if len(driver_races) < 10:
            return 0.0

        # Analyze performance in close battles (finished within 1 position of grid)
        clutch_scores = []
        for _, race in driver_races.iterrows():
            grid = race.get('grid', 0)
            finish = race.get('finish', 0)
            if grid > 0 and finish > 0:
                # Close battles: starting position was contested (not dominant)
                if 2 <= grid <= 10:  # Midfield or top battles
                    # Positive if outperformed grid position
                    score = (grid - finish) / 5  # Normalize
                    clutch_scores.append(score)

        if not clutch_scores:
            return 0.0
        avg = sum(clutch_scores) / len(clutch_scores)
        return max(-0.5, min(0.5, avg))

    def _calc_wet_performance(self, driver: str) -> float:
        """Performance delta in wet conditions vs dry.

        Positive = better in wet, negative = worse in wet.
        Range typically -1.0 to 1.0 (percentage improvement/degradation).
        """
        if self.laps.empty:
            return 0.0

        # Check for wet indicator in data
        if 'Rainfall' not in self.laps.columns and 'IsWet' not in self.laps.columns:
            return 0.0

        driver_laps = self.laps[self.laps['Driver'] == driver]
        if driver_laps.empty:
            return 0.0

        # Determine wet condition column
        wet_col = 'Rainfall' if 'Rainfall' in driver_laps.columns else 'IsWet'

        # Calculate average gap in wet vs dry
        wet_gaps = []
        dry_gaps = []

        for (season, circuit), session_laps in self.laps.groupby(['season', 'circuit']):
            driver_session = session_laps[session_laps['Driver'] == driver]
            if driver_session.empty:
                continue

            driver_time = driver_session['LapTime'].min()
            if pd.isna(driver_time):
                continue
            driver_sec = driver_time.total_seconds() if hasattr(driver_time, 'total_seconds') else float(driver_time)

            session_best = session_laps['LapTime'].min()
            if pd.isna(session_best):
                continue
            best_sec = session_best.total_seconds() if hasattr(session_best, 'total_seconds') else float(session_best)

            if best_sec <= 0:
                continue

            gap_pct = (driver_sec - best_sec) / best_sec * 100

            # Check if session was wet
            is_wet = session_laps[wet_col].any() if wet_col in session_laps.columns else False
            if is_wet:
                wet_gaps.append(gap_pct)
            else:
                dry_gaps.append(gap_pct)

        if not wet_gaps or not dry_gaps:
            return 0.0

        avg_wet = sum(wet_gaps) / len(wet_gaps)
        avg_dry = sum(dry_gaps) / len(dry_gaps)

        # Positive = smaller gap in wet (better wet performance)
        wet_advantage = avg_dry - avg_wet
        return max(-1.0, min(1.0, wet_advantage))

    def _calc_temp_sensitivity(self, driver: str) -> float:
        """Sensitivity to high track temperatures (performance degradation per degree).

        Returns coefficient: higher = more sensitive to heat.
        Typical range: 0.0001 to 0.005.
        """
        if self.laps.empty or 'TrackTemp' not in self.laps.columns:
            return 0.001  # Default mild sensitivity

        driver_laps = self.laps[self.laps['Driver'] == driver]
        if driver_laps.empty:
            return 0.001

        # Correlate track temp with performance gap
        temp_gaps = []
        for (season, circuit), session_laps in self.laps.groupby(['season', 'circuit']):
            driver_session = session_laps[session_laps['Driver'] == driver]
            if driver_session.empty:
                continue

            # Get track temp for this session
            temps = session_laps['TrackTemp'].dropna()
            if temps.empty:
                continue
            avg_temp = temps.mean()

            driver_time = driver_session['LapTime'].min()
            if pd.isna(driver_time):
                continue
            driver_sec = driver_time.total_seconds() if hasattr(driver_time, 'total_seconds') else float(driver_time)

            session_best = session_laps['LapTime'].min()
            if pd.isna(session_best):
                continue
            best_sec = session_best.total_seconds() if hasattr(session_best, 'total_seconds') else float(session_best)

            if best_sec > 0:
                gap_pct = (driver_sec - best_sec) / best_sec * 100
                temp_gaps.append((avg_temp, gap_pct))

        if len(temp_gaps) < 5:
            return 0.001

        # Simple correlation: does gap increase with temp?
        temps = np.array([t[0] for t in temp_gaps])
        gaps = np.array([t[1] for t in temp_gaps])

        if temps.std() < 5:  # Not enough temp variation
            return 0.001

        # Linear fit: sensitivity is slope
        try:
            slope = np.polyfit(temps, gaps, 1)[0]
            # Positive slope = worse at higher temps
            return max(0.0, min(0.005, slope / 100))
        except Exception:
            return 0.001
```

## 4. Model Layer

### Approach A: Elo/Glicko (interpretable)

```python
import numpy as np

class F1EloSystem:
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        self.k = k_factor
        self.initial = initial_rating
        self.ratings: dict = {}
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        # Clamp exponent to prevent overflow (diff of 4000 points = exponent of 10)
        exponent = max(-10, min(10, (rating_b - rating_a) / 400))
        return 1 / (1 + 10 ** exponent)
    
    def update_quali_ratings(self, quali_results: list[tuple[str, float]]):
        n = len(quali_results)
        if n < 2:
            # Need at least 2 drivers to compute pairwise ratings
            return

        # Initialize any new drivers first
        for driver, _ in quali_results:
            if driver not in self.ratings:
                self.ratings[driver] = {'quali': self.initial, 'race': self.initial}

        # Calculate all deltas using current ratings (before any updates)
        deltas = {}
        for i, (driver_a, time_a) in enumerate(quali_results):
            rating_a = self.ratings[driver_a]['quali']
            delta = 0
            for j, (driver_b, time_b) in enumerate(quali_results):
                if i == j:
                    continue
                rating_b = self.ratings[driver_b]['quali']
                expected = self.expected_score(rating_a, rating_b)
                # Handle ties: 0.5 for equal times, 1.0 for win, 0.0 for loss
                if time_a < time_b:
                    actual = 1.0
                elif time_a > time_b:
                    actual = 0.0
                else:
                    actual = 0.5  # tie
                delta += self.k * (actual - expected) / (n - 1)
            deltas[driver_a] = delta

        # Apply all deltas after calculation
        for driver, delta in deltas.items():
            self.ratings[driver]['quali'] += delta
    
    def update_race_ratings(self, race_results: list[tuple[str, int]]):
        """Update race Elo ratings based on finishing positions.

        Args:
            race_results: List of (driver, finish_position) tuples
        """
        n = len(race_results)
        if n < 2:
            return

        # Initialize any new drivers first
        for driver, _ in race_results:
            if driver not in self.ratings:
                self.ratings[driver] = {'quali': self.initial, 'race': self.initial}

        # Calculate all deltas using current ratings (before any updates)
        deltas = {}
        for i, (driver_a, pos_a) in enumerate(race_results):
            rating_a = self.ratings[driver_a]['race']
            delta = 0
            for j, (driver_b, pos_b) in enumerate(race_results):
                if i == j:
                    continue
                rating_b = self.ratings[driver_b]['race']
                expected = self.expected_score(rating_a, rating_b)
                # Lower position number = better result
                if pos_a < pos_b:
                    actual = 1.0
                elif pos_a > pos_b:
                    actual = 0.0
                else:
                    actual = 0.5  # tie (rare but possible)
                delta += self.k * (actual - expected) / (n - 1)
            deltas[driver_a] = delta

        # Apply all deltas after calculation
        for driver, delta in deltas.items():
            self.ratings[driver]['race'] += delta

    def predict_quali_probs(self, drivers: list[str]) -> dict[str, float]:
        if not drivers:
            return {}
        ratings = {d: self.ratings.get(d, {}).get('quali', self.initial) for d in drivers}
        # Use softmax with max subtraction for numerical stability
        # Scale factor 100: ~100 rating points difference = ~2.7x probability ratio
        # This provides reasonable spread without extreme probability differences
        ELO_SCALE_FACTOR = 100
        scaled = {d: r / ELO_SCALE_FACTOR for d, r in ratings.items()}
        max_scaled = max(scaled.values()) if scaled else 0
        exp_ratings = {d: np.exp(s - max_scaled) for d, s in scaled.items()}
        total = sum(exp_ratings.values())
        n = len(drivers)
        return {d: exp_r / total for d, exp_r in exp_ratings.items()} if total > 0 else {d: 1.0 / n for d in drivers}
```

### Approach B: Bayesian Hierarchical (PyMC/Stan)

```python
import pymc as pm
import numpy as np
import pandas as pd

def build_quali_model(lap_times: pd.DataFrame):
    # Filter out rows with invalid lap times (NaT)
    lap_times = lap_times.dropna(subset=['LapTime']).copy()
    if lap_times.empty:
        raise ValueError("No valid lap times to build model")

    teams = lap_times['Team'].unique()
    drivers = lap_times['Driver'].unique()
    circuits = lap_times['circuit'].unique()  # Added by feature engine

    # Calculate gap to fastest lap per session/circuit
    # Group by circuit to find the fastest lap, then compute gap percentage
    # Handle both timedelta and numeric LapTime columns
    if hasattr(lap_times['LapTime'].dtype, 'kind') and lap_times['LapTime'].dtype.kind == 'm':
        # timedelta64 type - use .dt accessor
        lap_times['LapTimeSeconds'] = lap_times['LapTime'].dt.total_seconds()
    else:
        # Already numeric (float/int) - convert directly
        lap_times['LapTimeSeconds'] = lap_times['LapTime'].apply(
            lambda x: x.total_seconds() if hasattr(x, 'total_seconds') else float(x)
        )
    fastest_per_circuit = lap_times.groupby('circuit')['LapTimeSeconds'].transform('min')
    lap_times['gap_to_fastest'] = (lap_times['LapTimeSeconds'] - fastest_per_circuit) / fastest_per_circuit * 100

    # Create mappings and validate no missing values
    team_map = {t: i for i, t in enumerate(teams)}
    driver_map = {d: i for i, d in enumerate(drivers)}
    circuit_map = {c: i for i, c in enumerate(circuits)}

    # Filter to only rows with known teams/drivers/circuits
    lap_times = lap_times[
        lap_times['Team'].isin(team_map) &
        lap_times['Driver'].isin(driver_map) &
        lap_times['circuit'].isin(circuit_map)
    ]
    if lap_times.empty:
        raise ValueError("No valid data after filtering unknown teams/drivers/circuits")

    team_idx = lap_times['Team'].map(team_map).astype(int).values
    driver_idx = lap_times['Driver'].map(driver_map).astype(int).values
    circuit_idx = lap_times['circuit'].map(circuit_map).astype(int).values
    with pm.Model() as quali_model:
        team_mu = pm.Normal('team_mu', mu=0, sigma=1)
        team_sigma = pm.HalfNormal('team_sigma', sigma=0.5)
        team_effect = pm.Normal('team_effect', mu=team_mu, sigma=team_sigma, shape=len(teams))
        driver_sigma = pm.HalfNormal('driver_sigma', sigma=0.3)
        driver_effect = pm.Normal('driver_effect', mu=0, sigma=driver_sigma, shape=len(drivers))
        circuit_sigma = pm.HalfNormal('circuit_sigma', sigma=0.2)
        circuit_effect = pm.Normal('circuit_effect', mu=0, sigma=circuit_sigma, shape=len(circuits))
        mu = team_effect[team_idx] + driver_effect[driver_idx] + circuit_effect[circuit_idx]
        sigma = pm.HalfNormal('sigma', sigma=0.1)
        pm.Normal('y', mu=mu, sigma=sigma, observed=lap_times['gap_to_fastest'].values)
        # PyMC 5.x returns InferenceData by default
        # Set random_seed for reproducibility
        idata = pm.sample(2000, tune=1000, cores=4, return_inferencedata=True, random_seed=42)
    return quali_model, idata
```

### Approach C: Gradient Boosting (LightGBM)

```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

def train_quali_model(features_df: pd.DataFrame, target: pd.Series):
    tscv = TimeSeriesSplit(n_splits=5)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }
    models = []
    for train_idx, val_idx in tscv.split(features_df):
        X_train, X_val = features_df.iloc[train_idx], features_df.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
            ],
        )
        models.append(model)
    return models
```

## 5. Race Simulation (Monte Carlo)

```python
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import random

@dataclass
class CarState:
    driver: str
    team: str
    position: int
    lap: int
    tire_compound: str
    tire_age: int
    fuel_load: float
    time_behind_leader: float
    pit_stops: int
    cumulative_time: float = 0.0
    drs_enabled: bool = False
    dnf: bool = False
    # Track compounds used (F1 requires 2 different dry compounds)
    # Using field(default_factory=set) avoids mutable default issues
    used_compounds: set = field(default_factory=set)
    # Track laps completed for lapped car handling during SC
    laps_completed: int = 0
    # Store last lap time for dirty air calculations
    last_lap_time: float = 0.0

    def __post_init__(self):
        # Always ensure starting compound is tracked
        # Even if used_compounds was pre-populated, add current compound
        self.used_compounds.add(self.tire_compound)

@dataclass
class RaceConfig:
    total_laps: int
    pit_loss: float  # seconds lost in pit stop
    overtake_delta: float  # pace delta needed to attempt overtake
    sc_probability: float  # per-lap safety car probability
    vsc_probability: float  # per-lap VSC probability
    red_flag_probability: float  # per-lap red flag probability
    dnf_rates: dict[str, float]  # team -> DNF probability per lap
    drs_zones: int  # number of DRS zones (informational)
    drs_delta: float  # total seconds gained per lap when DRS enabled (~0.2-0.4s)
    tire_compounds: dict[str, dict]  # compound -> {pace_delta, deg_rate}
    driver_teams: dict[str, str]  # driver -> team mapping
    # Dirty air parameters
    dirty_air_threshold: float = 2.0  # seconds - within this gap, dirty air affects pace
    dirty_air_penalty: float = 0.5  # seconds slower per lap when in dirty air

class RaceSimulator:
    def __init__(self, config: RaceConfig):
        self.config = config

    def run_monte_carlo(
        self,
        n_simulations: int,
        grid_probs: dict[str, list[float]],
        base_pace: dict[str, float],
        tire_deg: dict[str, float],
        driver_variance: dict[str, float],
        driver_dnf_rates: dict[str, float] | None = None,
        seed: int | None = None,
        track_condition: str = 'dry',
    ) -> dict[str, dict[int, float]]:
        """Run n simulations and return position probability distributions.

        Args:
            seed: Optional random seed for reproducibility. If None, results vary.
            track_condition: 'dry', 'damp' (intermediate), or 'wet' (full wet)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        results = defaultdict(lambda: defaultdict(int))
        driver_dnf_rates = driver_dnf_rates or {}

        for _ in range(n_simulations):
            # Sample grid from qualifying probabilities
            grid = self._sample_grid(grid_probs)

            # Run single race simulation
            race_result = self.simulate_race(
                grid, base_pace, tire_deg, driver_variance, driver_dnf_rates, track_condition
            )

            # Accumulate results
            for driver, position in race_result:
                results[driver][position] += 1

        # Convert counts to probabilities
        return {
            driver: {pos: count / n_simulations for pos, count in positions.items()}
            for driver, positions in results.items()
        }

    def _sample_grid(self, grid_probs: dict[str, list[float]]) -> list[str]:
        """Sample a starting grid from qualifying position probabilities.

        Always returns a grid with all drivers, even if probabilities are incomplete.
        """
        drivers = list(grid_probs.keys())
        if not drivers:
            return []

        grid = []
        remaining = set(drivers)

        for pos in range(len(drivers)):
            if not remaining:
                break  # All drivers placed

            # Safe index access with bounds check
            probs = [
                grid_probs[d][pos] if (d in remaining and pos < len(grid_probs.get(d, []))) else 0
                for d in drivers
            ]
            total = sum(probs)

            if total > 0:
                probs = [p / total for p in probs]
            else:
                # Fallback: uniform distribution over remaining drivers
                n_remaining = len(remaining)
                probs = [1.0 / n_remaining if d in remaining else 0 for d in drivers]

            # Ensure probs sum to exactly 1.0 (fix floating point errors)
            prob_sum = sum(probs)
            if prob_sum > 0 and abs(prob_sum - 1.0) > 1e-9:
                probs = [p / prob_sum for p in probs]

            selected = np.random.choice(drivers, p=probs)
            grid.append(selected)
            remaining.discard(selected)  # Use discard to avoid KeyError

        # Safety check: if any drivers were missed, append them
        for d in remaining:
            grid.append(d)

        return grid

    def simulate_race(
        self,
        grid: list[str],
        base_pace: dict[str, float],
        tire_deg: dict[str, float],
        driver_variance: dict[str, float],
        driver_dnf_rates: dict[str, float] | None = None,
        track_condition: str = 'dry',
    ) -> list[tuple[str, int]]:
        """Simulate a single race, returns list of (driver, position).

        Args:
            track_condition: 'dry', 'damp' (intermediate), or 'wet' (full wet)
        """
        driver_dnf_rates = driver_dnf_rates or {}
        is_wet = track_condition in ('damp', 'wet')  # For pit stop logic
        cars = self._initialize_cars(grid, track_condition)
        cars = self._simulate_lap_1(cars, base_pace, tire_deg, driver_variance)
        drs_disabled_until = 0  # Track when DRS is re-enabled after SC

        for lap in range(2, self.config.total_laps + 1):
            # Check for race-interrupting events
            if random.random() < self.config.red_flag_probability:
                cars = self._handle_red_flag(cars, lap, track_condition)
                drs_disabled_until = lap + 2  # DRS disabled for 2 laps after restart
            elif random.random() < self.config.sc_probability:
                cars = self._handle_safety_car(cars, lap)
                drs_disabled_until = lap + 2  # DRS disabled for 2 laps after SC
            elif random.random() < self.config.vsc_probability:
                cars = self._handle_vsc(cars, lap)
                drs_disabled_until = lap + 1  # DRS disabled for 1 lap after VSC

            # Sort cars by position for dirty air calculation
            sorted_cars = sorted([c for c in cars if not c.dnf], key=lambda x: x.cumulative_time)
            car_ahead_times = {}  # Map driver -> car ahead's last lap time
            for i, car in enumerate(sorted_cars):
                if i > 0:
                    car_ahead_times[car.driver] = sorted_cars[i - 1].last_lap_time

            # Simulate each car's lap
            for car in cars:
                if car.dnf:
                    continue
                # Use driver-specific DNF rate if available, otherwise use team rate
                dnf_rate = driver_dnf_rates.get(
                    car.driver,
                    self.config.dnf_rates.get(car.team, 0.002)
                )
                if random.random() < dnf_rate:
                    car.dnf = True
                    car.lap = lap  # Record lap of retirement for classification
                    continue

                # Calculate clean air lap time
                clean_air_time = self._calculate_lap_time(
                    car,
                    base_pace.get(car.driver, 90.0),
                    tire_deg.get(car.driver, 0.05),
                    driver_variance.get(car.driver, 0.15)
                )

                # Apply dirty air constraint if within threshold of car ahead
                lap_time = clean_air_time
                if car.time_behind_leader > 0:  # Not the leader
                    # Find gap to car directly ahead
                    car_ahead_lap = car_ahead_times.get(car.driver, 0)
                    if car_ahead_lap > 0 and car.time_behind_leader < self.config.dirty_air_threshold:
                        # In dirty air: add penalty and constrain by car ahead's pace
                        dirty_air_time = clean_air_time + self.config.dirty_air_penalty
                        # Can't physically go faster than the car ahead without overtaking
                        lap_time = max(dirty_air_time, car_ahead_lap)

                car.cumulative_time += lap_time
                car.last_lap_time = lap_time  # Store for next lap's dirty air calc
                car.tire_age += 1
                car.fuel_load = max(0, car.fuel_load - 1.5)  # Can't go below 0
                car.lap = lap
                car.laps_completed += 1

            cars = self._handle_pit_stops(cars, lap, track_condition, tire_deg)
            cars = self._simulate_overtakes(cars, base_pace, tire_deg)
            drs_disabled = lap <= drs_disabled_until
            cars = self._update_positions(cars, lap=lap, drs_disabled=drs_disabled)

        # Sort: active cars by position, then DNF cars by lap they retired (later = better)
        active = sorted([c for c in cars if not c.dnf], key=lambda x: x.cumulative_time)
        # DNF tie-breaker: if same lap, use cumulative_time (further into lap = better classification)
        dnf_cars = sorted([c for c in cars if c.dnf], key=lambda x: (x.lap, x.cumulative_time), reverse=True)

        # Assign final positions: active cars first, then DNFs
        results = []
        for i, car in enumerate(active):
            results.append((car.driver, i + 1))
        for i, car in enumerate(dnf_cars):
            results.append((car.driver, len(active) + i + 1))

        return results

    def _initialize_cars(self, grid: list[str], track_condition: str = 'dry') -> list[CarState]:
        """Initialize car states for race start.

        Args:
            grid: Starting grid order
            track_condition: 'dry', 'damp' (intermediate), or 'wet' (full wet)
        """
        def get_starting_tire(pos: int) -> str:
            if track_condition == 'wet':
                return 'WET'  # Heavy rain requires full wet tires
            elif track_condition == 'damp':
                return 'INTERMEDIATE'  # Light rain / drying track
            else:
                # Dry: Q3 participants (top 10) start on Q2 tires (usually SOFT)
                return 'SOFT' if pos < 10 else 'MEDIUM'

        return [
            CarState(
                driver=driver,
                team=self.config.driver_teams.get(driver, 'Unknown'),
                position=pos + 1,
                lap=0,
                tire_compound=get_starting_tire(pos),
                tire_age=0 if track_condition != 'dry' else (4 if pos < 10 else 0),  # Fresh tires in wet
                fuel_load=110.0,
                time_behind_leader=0.0,
                pit_stops=0,
            )
            for pos, driver in enumerate(grid)
        ]

    def _simulate_lap_1(
        self, cars: list[CarState], base_pace: dict, tire_deg: dict, driver_variance: dict
    ) -> list[CarState]:
        """Lap 1 has higher variance due to starts and turn 1 incidents.

        Lap 1 has ~3-5x higher incident rate than normal racing laps.
        """
        LAP_1_DNF_MULTIPLIER = 4.0  # Lap 1 incidents are ~4x more likely

        for car in cars:
            # Check for lap 1 incident (higher rate than normal laps)
            base_dnf_rate = self.config.dnf_rates.get(car.team, 0.002)
            if random.random() < base_dnf_rate * LAP_1_DNF_MULTIPLIER:
                car.dnf = True
                car.lap = 1
                continue
            # Calculate base lap time for lap 1
            base_lap_time = self._calculate_lap_time(
                car,
                base_pace.get(car.driver, 90.0),
                tire_deg.get(car.driver, 0.05),
                driver_variance.get(car.driver, 0.15)
            )
            # Start performance: random gain/loss of positions (higher variance on lap 1)
            # Positive = positions gained (faster, less time), negative = positions lost
            start_delta = np.random.normal(0, 1.5)  # positions gained/lost
            lap_time = base_lap_time - start_delta * 0.5  # Subtract: gaining positions = less time
            car.cumulative_time += lap_time
            car.tire_age += 1
            car.fuel_load = max(0, car.fuel_load - 1.5)  # Can't go below 0
            car.lap = 1
        return self._update_positions(cars, lap=1, drs_disabled=True)  # No DRS on lap 1

    def _calculate_lap_time(
        self, car: CarState, base: float, deg: float, variance: float
    ) -> float:
        """Calculate lap time with tire deg, fuel effect, and random variance."""
        compound_info = self.config.tire_compounds.get(car.tire_compound, {})
        # Use compound degradation rate as base, adjusted by driver factor
        # Driver deg is relative to MEDIUM (0.05); > 0.05 means driver degrades tires faster
        compound_deg = compound_info.get('deg_rate', 0.05)
        driver_factor = deg / 0.05 if deg > 0 else 1.0  # How driver compares to average
        effective_deg = compound_deg * driver_factor
        tire_effect = car.tire_age * effective_deg
        fuel_effect = (110.0 - car.fuel_load) * 0.03  # lighter = faster
        compound_delta = compound_info.get('pace_delta', 0)

        # DRS gives ~0.3s per activation, more zones = more opportunities but diminishing returns
        # Typical gain: 0.2-0.4s total per lap depending on circuit
        drs_gain = self.config.drs_delta if car.drs_enabled else 0
        noise = np.random.normal(0, variance)

        return base + tire_effect - fuel_effect + compound_delta - drs_gain + noise

    def _handle_safety_car(self, cars: list[CarState], lap: int) -> list[CarState]:
        """Bunch up the field during safety car.

        Race order is preserved based on cumulative time (whoever has less time
        is ahead). Gaps are compressed to 0.5s per position, simulating the
        bunching effect of a safety car period. Tire degradation is reduced
        due to slower pace under SC.

        Lapped cars maintain their lapped status - they don't get a "free" unlap
        unless explicitly allowed (which requires being in front of the SC).
        """
        # Under SC, tires degrade at ~50% normal rate due to slower pace
        # Model this by not incrementing tire_age during SC laps
        active = [c for c in cars if not c.dnf]
        if not active:
            return cars

        # Sort by cumulative_time to get accurate current positions
        # Note: Python sort is stable, so cars with equal times keep their order
        active.sort(key=lambda x: x.cumulative_time)
        leader = active[0]
        leader_time = leader.cumulative_time
        leader_laps = leader.laps_completed

        # Group cars by laps completed (to preserve lapped status)
        for i, car in enumerate(active):
            laps_down = leader_laps - car.laps_completed
            if laps_down <= 0:
                # On the lead lap - compress gaps normally
                car.cumulative_time = leader_time + i * 0.5
            else:
                # Lapped car - they bunch up but remain lapped
                # Their cumulative time reflects being laps_down behind
                lap_time_estimate = 90.0  # Approximate lap time for spacing
                car.cumulative_time = leader_time + (laps_down * lap_time_estimate) + i * 0.5

            car.time_behind_leader = car.cumulative_time - leader_time
            # Don't increment tire_age this lap (SC pace = minimal degradation)
            # Note: tire_age was already incremented in main loop, so decrement by 1
            # to effectively skip degradation for this SC lap
            car.tire_age = max(0, car.tire_age - 1)
        return cars

    def _handle_vsc(self, cars: list[CarState], lap: int) -> list[CarState]:
        """VSC: gaps maintained but reduced by 20%, tire degradation slightly reduced."""
        active = [c for c in cars if not c.dnf]
        if not active:
            return cars
        # Sort by cumulative_time to get accurate ordering
        active.sort(key=lambda x: x.cumulative_time)
        leader_time = active[0].cumulative_time
        for car in active:
            gap = car.cumulative_time - leader_time
            car.cumulative_time = leader_time + gap * 0.8
            car.time_behind_leader = car.cumulative_time - leader_time
        # VSC has less impact on tire deg than full SC, but still reduces it slightly
        # Only reduce tire age for ~1/3 of cars (random selection to model uncertainty)
        if random.random() < 0.3:
            for car in active:
                car.tire_age = max(0, car.tire_age - 1)
        return cars

    def _handle_red_flag(
        self, cars: list[CarState], lap: int, track_condition: str = 'dry'
    ) -> list[CarState]:
        """Red flag: reset gaps to standing start intervals, free tire change.

        Args:
            track_condition: Current track condition ('dry', 'damp', 'wet').
                            May differ from race start if conditions changed.
        """
        active = [c for c in cars if not c.dnf]
        if not active:
            return cars
        # Sort by cumulative_time to get accurate current positions
        active.sort(key=lambda x: x.cumulative_time)
        leader_time = active[0].cumulative_time
        remaining_laps = self.config.total_laps - lap

        for i, car in enumerate(active):
            # Reset to standing start gaps
            car.cumulative_time = leader_time + i * 0.1
            car.time_behind_leader = car.cumulative_time - leader_time
            car.tire_age = 0  # free tire change
            # Choose optimal compound based on current conditions and remaining distance
            if track_condition == 'wet':
                car.tire_compound = 'WET'
            elif track_condition == 'damp':
                car.tire_compound = 'INTERMEDIATE'
            elif remaining_laps > 30:
                car.tire_compound = 'HARD'  # Long stint needs durability
            elif remaining_laps > 15:
                car.tire_compound = 'MEDIUM'
            else:
                car.tire_compound = 'SOFT'  # Sprint to the end
            car.used_compounds.add(car.tire_compound)  # Track for 2-compound rule
        return cars

    def _handle_pit_stops(
        self, cars: list[CarState], lap: int, track_condition: str = 'dry',
        tire_deg: dict | None = None
    ) -> list[CarState]:
        """Pit strategy based on tire compound optimal laps and race distance.

        Args:
            track_condition: 'dry', 'damp' (intermediate), or 'wet' (full wet)
            tire_deg: Dict of driver tire degradation rates for smarter pit decisions

        Enforces F1's mandatory 2-compound rule for dry races.
        """
        remaining_laps = self.config.total_laps - lap
        dry_compounds = {'SOFT', 'MEDIUM', 'HARD'}
        is_wet = track_condition in ('damp', 'wet')
        tire_deg = tire_deg or {}

        for car in cars:
            if car.dnf:
                continue
            # Get optimal laps for current compound from config
            compound_info = self.config.tire_compounds.get(car.tire_compound, {})
            optimal_laps = compound_info.get('optimal_laps', 30)

            # Adjust optimal laps based on driver's tire degradation rate
            driver_deg = tire_deg.get(car.driver_id, 0.0)
            if driver_deg > 0.05:  # High degradation driver
                optimal_laps = int(optimal_laps * 0.85)  # Pit earlier
            elif driver_deg < 0.02:  # Tire whisperer
                optimal_laps = int(optimal_laps * 1.1)  # Can extend stint

            # Pit if tire age exceeds optimal and enough laps remain to benefit
            if car.tire_age > optimal_laps and remaining_laps > 5:
                car.cumulative_time += self.config.pit_loss

                # Choose compound based on conditions and remaining laps
                if track_condition == 'wet':
                    new_compound = 'WET'
                elif track_condition == 'damp':
                    new_compound = 'INTERMEDIATE'
                elif remaining_laps > 30:
                    new_compound = 'HARD'
                elif remaining_laps > 15:
                    new_compound = 'MEDIUM'
                else:
                    new_compound = 'SOFT'

                # Enforce 2-compound rule: if only used one dry compound, must use different
                used_dry = car.used_compounds & dry_compounds
                if len(used_dry) == 1 and new_compound in used_dry and not is_wet:
                    # Must choose a different compound
                    available = dry_compounds - used_dry
                    if remaining_laps > 20:
                        new_compound = 'MEDIUM' if 'MEDIUM' in available else available.pop()
                    else:
                        new_compound = 'SOFT' if 'SOFT' in available else available.pop()

                car.tire_compound = new_compound
                car.used_compounds.add(new_compound)
                car.tire_age = 0
                car.pit_stops += 1
        return cars

    def _simulate_overtakes(
        self, cars: list[CarState], base_pace: dict, tire_deg: dict
    ) -> list[CarState]:
        """Attempt overtakes based on pace delta.

        Uses multiple passes to handle cascading overtakes properly.
        """
        max_passes = 3  # Limit passes to avoid infinite loops
        for _ in range(max_passes):
            overtake_occurred = False
            sorted_cars = sorted(cars, key=lambda x: x.cumulative_time)

            for i in range(1, len(sorted_cars)):
                car_behind = sorted_cars[i]
                car_ahead = sorted_cars[i - 1]
                if car_behind.dnf or car_ahead.dnf:
                    continue

                pace_behind = base_pace.get(car_behind.driver, 90) + car_behind.tire_age * tire_deg.get(car_behind.driver, 0.05)
                pace_ahead = base_pace.get(car_ahead.driver, 90) + car_ahead.tire_age * tire_deg.get(car_ahead.driver, 0.05)
                pace_delta = pace_ahead - pace_behind

                # DRS boost for car behind
                if car_behind.drs_enabled:
                    pace_delta += self.config.drs_delta

                if pace_delta > self.config.overtake_delta:
                    overtake_prob = min(0.5, pace_delta / 2.0)
                    if random.random() < overtake_prob:
                        # Swap positions: car_behind passes car_ahead
                        # After overtake, the new leader is ~0.3s ahead (realistic gap)
                        # Ensure cumulative_time stays positive (can't have negative race time)
                        new_behind_time = max(0.1, car_ahead.cumulative_time - 0.1)
                        car_behind.cumulative_time = new_behind_time
                        car_ahead.cumulative_time = new_behind_time + 0.3
                        overtake_occurred = True

            if not overtake_occurred:
                break  # No more overtakes possible

        return cars

    def _update_positions(
        self, cars: list[CarState], lap: int = 3, drs_disabled: bool = False
    ) -> list[CarState]:
        """Update positions based on cumulative time.

        Args:
            cars: List of car states
            lap: Current lap number (DRS disabled for first 2 laps)
            drs_disabled: True if DRS is disabled (e.g., after SC restart)
        """
        active = [c for c in cars if not c.dnf]
        active.sort(key=lambda x: x.cumulative_time)
        for i, car in enumerate(active):
            car.position = i + 1
            car.time_behind_leader = car.cumulative_time - active[0].cumulative_time
            # DRS disabled for first 2 laps, after SC/VSC, and for leader
            if lap <= 2 or drs_disabled or i == 0:
                car.drs_enabled = False
            else:
                # DRS enabled if within 1 second of car directly ahead
                gap_to_car_ahead = car.cumulative_time - active[i - 1].cumulative_time
                car.drs_enabled = gap_to_car_ahead < 1.0
        return cars
```

## 6. Putting It All Together

```python
import numpy as np
import pandas as pd
from src.config import DRIVER_TEAMS, TIRE_COMPOUNDS, DEFAULT_DNF_RATES, CIRCUITS

class F1Predictor:
    def __init__(self):
        self.data_loader = F1DataLoader()
        self.feature_engine = F1FeatureEngine(self.data_loader)
        self.elo_system = F1EloSystem()
        self._processed_seasons: set[int] = set()  # Track processed seasons to avoid duplicate Elo updates
        self._features_loaded: bool = False  # Track if feature engine has historical data

    def _get_circuit_info(self, race: str) -> dict:
        """Look up circuit info from CIRCUITS config, with fallback defaults.

        Handles FastF1 event names like 'Bahrain Grand Prix' by extracting
        the location name ('Bahrain') for lookup in CIRCUITS.
        """
        # Direct lookup first
        if race in CIRCUITS:
            return CIRCUITS[race]

        # Try to extract location from full event name
        # FastF1 uses names like 'Bahrain Grand Prix', 'Monaco Grand Prix'
        race_lower = race.lower()
        for circuit_name in CIRCUITS:
            if circuit_name.lower() in race_lower:
                return CIRCUITS[circuit_name]

        # Fallback defaults
        return {
            'laps': 58,
            'pit_loss': 22.0,
            'drs_zones': 2,
            'overtake_delta': 0.8,
        }

    def _create_race_config(
        self, circuit_info: dict, tire_compounds: dict | None = None
    ) -> RaceConfig:
        """Create race config with circuit-specific parameters.

        Args:
            circuit_info: Circuit parameters (laps, pit_loss, etc.)
            tire_compounds: Optional dynamic tire compounds from practice data.
                           Falls back to default TIRE_COMPOUNDS if not provided.
        """
        return RaceConfig(
            total_laps=circuit_info.get('laps', 58),
            pit_loss=circuit_info.get('pit_loss', 22.0),
            overtake_delta=circuit_info.get('overtake_delta', 0.8),
            sc_probability=0.01,
            vsc_probability=0.015,
            red_flag_probability=0.002,
            dnf_rates=DEFAULT_DNF_RATES,
            drs_zones=circuit_info.get('drs_zones', 2),
            drs_delta=0.3,
            tire_compounds=tire_compounds or TIRE_COMPOUNDS,
            driver_teams=DRIVER_TEAMS,
        )

    def apply_grid_penalties(
        self, quali_positions: dict[str, int], penalties: dict[str, int | str]
    ) -> dict[str, int]:
        """Apply grid penalties (engine, gearbox, etc.) to qualifying results.

        Args:
            quali_positions: Driver -> qualifying position
            penalties: Driver -> penalty positions (int) or penalty type name (str)
                       Valid penalty types from PENALTY_TYPES: 'engine', 'full_pu',
                       'gearbox', 'pitlane_start'
        """
        from src.config import PENALTY_TYPES

        # Convert penalty type names to position values
        resolved_penalties = {}
        for driver, penalty in penalties.items():
            if isinstance(penalty, str):
                resolved_penalties[driver] = PENALTY_TYPES.get(penalty, 0)
            else:
                resolved_penalties[driver] = penalty

        # Sort by quali position
        sorted_drivers = sorted(quali_positions.items(), key=lambda x: x[1])
        penalized = [(d, pos + resolved_penalties.get(d, 0), pos) for d, pos in sorted_drivers]
        # Re-sort with penalties applied; use original quali position as tie-breaker
        # This ensures consistent ordering when multiple drivers have same penalized position
        penalized.sort(key=lambda x: (x[1], x[2]))
        # Assign final grid positions
        return {d: i + 1 for i, (d, _, _) in enumerate(penalized)}

    def predict_weekend(
        self,
        season: int,
        race: str,
        grid_penalties: dict[str, int | str] | None = None,
        circuit_info: dict | None = None,
        prediction_point: str = 'fp2',
        actual_grid: dict[str, int] | None = None,
    ) -> dict:
        """Predict qualifying and race outcomes for a weekend.

        Args:
            grid_penalties: Driver -> penalty positions (int) or penalty type name (str).
                           Valid types: 'engine', 'full_pu', 'gearbox', 'pitlane_start'
            prediction_point: When in the weekend we're predicting from:
                - 'fp1': After FP1 only (least data, highest uncertainty)
                - 'fp2': After FP2 (default - good balance of data availability)
                - 'fp3': After FP3 (most practice data)
                - 'quali': After qualifying (use actual_grid for race prediction only)
                - 'sprint': After sprint race (sprint weekend, use actual_grid)
            actual_grid: Dict of driver -> grid position. Required when prediction_point
                        is 'quali' or 'sprint'. Overrides quali prediction with actual results.
        """
        grid_penalties = grid_penalties or {}
        # Auto-lookup circuit info if not provided
        circuit_info = circuit_info or self._get_circuit_info(race)

        # Load historical data from PREVIOUS seasons only (avoid current season leakage)
        # Current season Elo updates should be done incrementally after each race
        historical_seasons = [s for s in range(season - 3, season) if s > 2017]
        for prev_season in historical_seasons:
            if prev_season not in self._processed_seasons:
                try:
                    historical = self.data_loader.load_season_data(prev_season)
                    # Update quali Elo with both main qualifying and sprint shootout
                    for quali_result in historical['qualifying']:
                        self.elo_system.update_quali_ratings(quali_result)
                    for sq_result in historical.get('sprint_qualifying', []):
                        self.elo_system.update_quali_ratings(sq_result)
                    # Update race Elo with both main race and sprint race
                    for race_result in historical['races']:
                        self.elo_system.update_race_ratings(race_result)
                    for sprint_result in historical.get('sprints', []):
                        self.elo_system.update_race_ratings(sprint_result)
                    self._processed_seasons.add(prev_season)
                except Exception:
                    pass  # Skip seasons with no data

        # Load feature engine historical data (once)
        if not self._features_loaded and historical_seasons:
            self.feature_engine.load_historical_data(historical_seasons)
            self._features_loaded = True

        # Load practice data based on prediction_point
        # Earlier prediction points have less data but allow earlier predictions
        fp_data = pd.DataFrame()
        available_sessions = {
            'fp1': ['FP1'],
            'fp2': ['FP2', 'FP1'],  # Prefer FP2, fall back to FP1
            'fp3': ['FP3', 'FP2', 'FP1'],  # Prefer FP3, then FP2, then FP1
            'quali': ['FP3', 'FP2', 'FP1'],  # All practice data available
            'sprint': ['FP3', 'FP2', 'FP1'],  # Sprint weekend - all practice
        }
        sessions_to_try = available_sessions.get(prediction_point, ['FP2', 'FP3', 'FP1'])

        for session in sessions_to_try:
            fp_data = self.data_loader.load_session(season, race, session)
            if not fp_data.empty:
                break

        weather = self.data_loader.get_weather(season, race, 'R')

        if fp_data.empty:
            raise ValueError(f"No practice data available for {season} {race}")

        drivers = fp_data['Driver'].unique().tolist()

        # Use actual grid if provided (post-quali prediction), otherwise predict
        if actual_grid and prediction_point in ('quali', 'sprint'):
            # Convert actual grid to deterministic probabilities (100% at actual position)
            n_drivers = len(drivers)
            quali_probs = {}
            for d in drivers:
                probs = [0.0] * n_drivers
                if d in actual_grid:
                    pos = actual_grid[d] - 1  # Convert 1-indexed to 0-indexed
                    if 0 <= pos < n_drivers:
                        probs[pos] = 1.0
                    else:
                        # Driver with grid penalty beyond field (pit lane start)
                        probs[-1] = 1.0
                else:
                    # Driver not in grid (DNS) - put at back
                    probs[-1] = 1.0
                quali_probs[d] = probs

            # Apply grid penalties to actual grid
            if grid_penalties:
                quali_probs = self._adjust_for_penalties(quali_probs, grid_penalties)
        else:
            # Predict qualifying from features
            quali_features = {
                d: self.feature_engine.calculate_quali_features(d, race)
                for d in drivers
            }
            quali_probs = self._predict_quali(drivers, quali_features)

            # Apply grid penalties to qualifying probabilities
            if grid_penalties:
                quali_probs = self._adjust_for_penalties(quali_probs, grid_penalties)

        # Extract race parameters
        base_pace = self._extract_race_pace(fp_data)
        tire_deg = self._extract_tire_deg(fp_data)
        # Extract circuit-specific tire compound deltas from practice data
        dynamic_tire_compounds = self._extract_tire_compound_deltas(fp_data)

        # Calculate race features to adjust driver variance
        race_features = {
            d: self.feature_engine.calculate_race_features(d, race, weather)
            for d in drivers
        }
        # Use clutch_factor to adjust variance (higher clutch = more consistent)
        # Clamp to [0.05, 0.25] to ensure valid variance range
        driver_variance = {
            d: max(0.05, min(0.25, 0.15 * (1 - race_features[d].get('clutch_factor', 0) * 0.2)))
            for d in drivers
        }

        # Increase variance for earlier prediction points (less data = more uncertainty)
        uncertainty_multiplier = {
            'fp1': 1.5,   # Highest uncertainty - only FP1 data
            'fp2': 1.2,   # Moderate uncertainty
            'fp3': 1.0,   # Normal uncertainty - all practice data
            'quali': 0.9,  # Lower uncertainty - actual grid known
            'sprint': 0.85,  # Lowest uncertainty - sprint result also known
        }.get(prediction_point, 1.0)

        driver_variance = {
            d: min(0.3, v * uncertainty_multiplier)  # Cap at 0.3 to avoid extreme variance
            for d, v in driver_variance.items()
        }

        # Create race config once and reuse (with dynamic tire compounds)
        race_config = self._create_race_config(circuit_info, dynamic_tire_compounds)

        # Extract driver-specific DNF rates from features (per-lap probability)
        driver_dnf_rates = {
            d: race_features[d].get('dnf_probability', 0.05) / race_config.total_laps
            for d in drivers
        }

        # Run Monte Carlo simulation
        simulator = RaceSimulator(race_config)

        # Determine track condition from weather
        # Note: More sophisticated logic could use rainfall intensity if available
        track_condition = 'damp' if weather.get('rainfall', False) else 'dry'

        race_probs = simulator.run_monte_carlo(
            n_simulations=10000,
            grid_probs=quali_probs,
            base_pace=base_pace,
            tire_deg=tire_deg,
            driver_variance=driver_variance,
            driver_dnf_rates=driver_dnf_rates,
            track_condition=track_condition,
        )

        # Confidence level based on data availability
        confidence_levels = {
            'fp1': 'low',
            'fp2': 'moderate',
            'fp3': 'good',
            'quali': 'high',
            'sprint': 'high',
        }

        return {
            'pole_probabilities': {
                d: quali_probs[d][0] if (quali_probs.get(d) and len(quali_probs[d]) > 0) else 1.0 / max(1, len(drivers))
                for d in drivers
            },
            'win_probabilities': {
                d: race_probs.get(d, {}).get(1, 0) for d in drivers
            },
            'podium_probabilities': {
                d: sum(race_probs.get(d, {}).get(p, 0) for p in [1, 2, 3])
                for d in drivers
            },
            'full_distributions': race_probs,
            'weather': weather,
            'prediction_point': prediction_point,
            'confidence': confidence_levels.get(prediction_point, 'moderate'),
            'grid_is_actual': actual_grid is not None and prediction_point in ('quali', 'sprint'),
        }

    def _predict_quali(
        self, drivers: list[str], features: dict
    ) -> dict[str, list[float]]:
        """Predict qualifying position probabilities for each driver."""
        if not drivers:
            return {}

        # Combine Elo ratings with features
        elo_probs = self.elo_system.predict_quali_probs(drivers)

        # Return position probability distribution per driver
        # Position 0 = P1, Position 1 = P2, etc.
        n = len(drivers)
        result = {}
        for driver in drivers:
            base_prob = elo_probs.get(driver, 1 / n)
            # Adjust base probability using features (form score and circuit affinity)
            driver_features = features.get(driver, {})
            form_adj = driver_features.get('form_score', 0) * 0.1  # form contributes up to 10%
            circuit_adj = driver_features.get('circuit_affinity', 0) * 0.05  # circuit adds 5%
            adjusted_prob = base_prob * (1 + form_adj + circuit_adj)
            # Clamp to valid probability range
            adjusted_prob = max(0.001, min(0.999, adjusted_prob))

            # Simple distribution around expected position
            probs = []
            # Sigma controls spread of position distribution
            # Use max(1, n/4) to ensure minimum spread for small grids
            sigma = max(1.0, n / 4)
            # Expected position: high prob = low position (front), low prob = high position (back)
            expected_pos = (1 - adjusted_prob) * n
            for pos in range(n):
                # Gaussian-like distribution centered on expected position
                prob = np.exp(-((pos - expected_pos) ** 2) / (2 * sigma ** 2))
                probs.append(prob)
            # Normalize (with fallback to uniform if all probs are ~0)
            total = sum(probs)
            if total > 0:
                result[driver] = [p / total for p in probs]
            else:
                result[driver] = [1.0 / n] * n  # uniform fallback
        return result

    def _adjust_for_penalties(
        self, quali_probs: dict, penalties: dict[str, int | str]
    ) -> dict[str, list[float]]:
        """Shift probability distributions for penalized drivers."""
        from src.config import PENALTY_TYPES

        adjusted = {}
        for driver, probs in quali_probs.items():
            raw_penalty = penalties.get(driver, 0)
            # Convert string penalty type to int positions
            if isinstance(raw_penalty, str):
                penalty = PENALTY_TYPES.get(raw_penalty, 0)
            else:
                penalty = raw_penalty
            if penalty > 0 and len(probs) > 0:
                n = len(probs)
                # For large penalties (back of grid), concentrate probability at the back
                if penalty >= n:
                    shifted = [0.0] * (n - 1) + [1.0]
                else:
                    # Shift distribution towards back of grid by penalty positions
                    # Positions 0 to penalty-1 become 0 (can't start there with penalty)
                    # Original position i becomes position i + penalty
                    shifted = [0.0] * n
                    for i, p in enumerate(probs):
                        new_pos = min(i + penalty, n - 1)
                        shifted[new_pos] += p
                adjusted[driver] = shifted
            else:
                adjusted[driver] = probs
        return adjusted

    def _extract_race_pace(self, fp_data) -> dict[str, float]:
        """Extract base race pace from practice data.

        Uses long run stints (5+ consecutive laps) to estimate true race pace,
        filtering out outliers from quali simulations and pit in/out laps.
        Uses explicit PitInTime/PitOutTime columns when available for accurate filtering.
        """
        DEFAULT_PACE = 90.0  # Fallback ~1:30 lap time in seconds

        if fp_data.empty:
            return {}

        # Filter out pit in/out laps using explicit columns if available
        filtered_data = fp_data.copy()
        if 'PitInTime' in filtered_data.columns:
            # Lap where driver pits (PitInTime is set) - exclude these
            filtered_data = filtered_data[filtered_data['PitInTime'].isna()]
        if 'PitOutTime' in filtered_data.columns:
            # Lap immediately after pit (PitOutTime is set) - exclude these
            filtered_data = filtered_data[filtered_data['PitOutTime'].isna()]

        # Filter to long run stints (consecutive laps on same compound)
        # This gives more representative race pace than single-lap pace
        def get_long_run_pace(driver_laps):
            if driver_laps.empty or 'LapTime' not in driver_laps.columns:
                return np.nan

            # Sort by lap number if available
            if 'LapNumber' in driver_laps.columns:
                driver_laps = driver_laps.sort_values('LapNumber')

            valid = driver_laps['LapTime'].dropna()
            if len(valid) < 3:
                # Not enough laps for long run analysis, use simple 25th percentile
                if valid.empty:
                    return np.nan
                q = valid.quantile(0.25)
                if pd.isna(q):
                    return np.nan
                return q.total_seconds() if hasattr(q, 'total_seconds') else float(q)

            # Use median of middle portion (laps 2-N-1) to exclude outliers
            # This approximates "long run" pace without complex stint detection
            times = valid.apply(lambda x: x.total_seconds() if hasattr(x, 'total_seconds') else float(x))
            middle_times = times.iloc[1:-1] if len(times) > 3 else times
            return middle_times.median() if not middle_times.empty else np.nan

        # Note: .apply() with include_groups=False avoids pandas FutureWarning
        pace = filtered_data.groupby('Driver').apply(get_long_run_pace, include_groups=False)
        # Filter out NaN values and provide default for missing drivers
        result = {d: p for d, p in pace.items() if pd.notna(p)}

        # Default pace for drivers without valid times
        drivers = fp_data['Driver'].unique()
        if result:
            default_pace = np.median(list(result.values()))
        else:
            # All drivers have invalid times, use fallback
            default_pace = DEFAULT_PACE

        for driver in drivers:
            if driver not in result:
                result[driver] = default_pace

        return result

    def _extract_tire_deg(self, fp_data) -> dict[str, float]:
        """Estimate tire degradation rate from long runs.

        Analyzes lap time progression during practice stints to estimate
        tire degradation (seconds lost per lap due to tire wear).
        """
        DEFAULT_DEG = 0.05  # Default degradation rate (seconds per lap)
        if fp_data.empty:
            return {}

        result = {}
        for driver in fp_data['Driver'].unique():
            driver_laps = fp_data[fp_data['Driver'] == driver].copy()

            # Need multiple laps to estimate degradation
            if len(driver_laps) < 5 or 'LapTime' not in driver_laps.columns:
                result[driver] = DEFAULT_DEG
                continue

            # Sort by lap number if available
            if 'LapNumber' in driver_laps.columns:
                driver_laps = driver_laps.sort_values('LapNumber')

            # Get valid lap times
            valid_times = driver_laps['LapTime'].dropna()
            if len(valid_times) < 5:
                result[driver] = DEFAULT_DEG
                continue

            # Convert to seconds
            times_sec = valid_times.apply(
                lambda x: x.total_seconds() if hasattr(x, 'total_seconds') else float(x)
            ).values

            # Simple linear regression to estimate degradation slope
            x = np.arange(len(times_sec))
            try:
                slope, _ = np.polyfit(x, times_sec, 1)
                # Clamp to reasonable range (0.01 to 0.15 sec/lap)
                # Negative slope means getting faster (warming up), use default
                result[driver] = max(0.01, min(0.15, slope)) if slope > 0 else DEFAULT_DEG
            except (np.linalg.LinAlgError, ValueError):
                result[driver] = DEFAULT_DEG

        # Fill missing drivers with default
        for driver in fp_data['Driver'].unique():
            if driver not in result:
                result[driver] = DEFAULT_DEG

        return result

    def _extract_tire_compound_deltas(self, fp_data) -> dict[str, dict]:
        """Extract actual tire compound pace deltas from practice data.

        Instead of using hardcoded compound deltas, analyze FP2 data to find
        the actual pace difference between compounds at this specific circuit.
        Returns updated tire_compounds dict with circuit-specific pace_deltas.
        """
        from src.config import TIRE_COMPOUNDS

        # Start with default compound info
        result = {compound: dict(info) for compound, info in TIRE_COMPOUNDS.items()}

        if fp_data.empty or 'Compound' not in fp_data.columns:
            return result

        # Get median pace per compound (filter outliers)
        compound_paces = {}
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            compound_laps = fp_data[fp_data['Compound'].str.upper() == compound]
            if compound_laps.empty or 'LapTime' not in compound_laps.columns:
                continue

            valid_times = compound_laps['LapTime'].dropna()
            if len(valid_times) < 3:
                continue

            # Convert to seconds and filter outliers (within 107% of best)
            times_sec = valid_times.apply(
                lambda x: x.total_seconds() if hasattr(x, 'total_seconds') else float(x)
            )
            best = times_sec.min()
            representative = times_sec[times_sec < best * 1.07]
            if not representative.empty:
                compound_paces[compound] = representative.median()

        # Calculate deltas relative to MEDIUM (baseline = 0)
        if 'MEDIUM' in compound_paces:
            medium_pace = compound_paces['MEDIUM']
            for compound, pace in compound_paces.items():
                # Negative delta = faster than medium, positive = slower
                delta = pace - medium_pace
                result[compound]['pace_delta'] = round(delta, 2)

        return result
```

## 7. Technology Stack Summary

| Component | Recommended | Alternative |
|-----------|-------------|-------------|
| Language | Python 3.11+ | R (stats) |
| Data Ingestion | FastF1, requests | httpx (async) |
| Database | DuckDB | PostgreSQL, SQLite |
| Data Processing | Pandas, Polars | - |
| Statistical Modeling | PyMC, statsmodels | Stan, brms (R) |
| ML Models | LightGBM, scikit-learn | XGBoost, CatBoost |
| Simulation | NumPy, numba | - |
| Visualization | Plotly, matplotlib | Altair |
| Dashboard | Streamlit, Gradio | Dash, Shiny |
| Deployment | Docker + FastAPI | AWS Lambda |

## 8. Quick Start Project Structure

```text
f1-predictor/
├── data/
│   ├── raw/
│   ├── processed/
│   └── f1_data.db
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── schema.py
│   ├── features/
│   │   └── engine.py
│   ├── models/
│   │   ├── elo.py
│   │   ├── bayesian.py
│   │   └── ml.py
│   ├── simulation/
│   │   └── race.py
│   ├── config.py
│   └── predictor.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_validation.ipynb
├── app/
│   └── streamlit_app.py
├── tests/
├── pyproject.toml
└── README.md
```

## 9. Validation Framework

```python
import fastf1
import numpy as np
from sklearn.calibration import calibration_curve

def get_races(season: int) -> list[str]:
    """Get list of race names for a season (past events only)."""
    from datetime import datetime, timezone
    import pandas as pd

    schedule = fastf1.get_event_schedule(season)
    today = datetime.now(timezone.utc)
    races = []
    for _, event in schedule.iterrows():
        if event['EventFormat'] == 'testing':
            continue
        # Filter out future events
        event_date = pd.to_datetime(event.get('EventDate', event.get('Session5Date')))
        if pd.notna(event_date):
            if event_date.tzinfo is None:
                event_date = event_date.tz_localize('UTC')
            if event_date > today:
                continue
        races.append(event['EventName'])
    return races

def get_actual_results(season: int, race: str) -> dict:
    """Get actual race results for validation and Elo updates."""
    pole = None
    winner = None
    podium = []
    quali_results = []  # For Elo updates
    race_results = []   # For Elo updates

    # Get qualifying results for pole position
    try:
        q_sess = fastf1.get_session(season, race, 'Q')
        q_sess.load()
        q_results = q_sess.results
        if not q_results.empty:
            # Filter out rows with NaN Position before casting to int
            q_valid = q_results.dropna(subset=['Position']).copy()
            if not q_valid.empty:
                q_valid['Position'] = q_valid['Position'].astype(int)
                pole_row = q_valid[q_valid['Position'] == 1]
                pole = pole_row['Abbreviation'].iloc[0] if not pole_row.empty else None
                # Extract ordered results for Elo updates
                quali_results = q_valid.sort_values('Position')['Abbreviation'].tolist()
    except Exception:
        pass  # Quali data unavailable

    # Get race results for winner and podium
    try:
        r_sess = fastf1.get_session(season, race, 'R')
        r_sess.load()
        r_results = r_sess.results
        if not r_results.empty:
            # Filter out rows with NaN Position before casting to int
            r_valid = r_results.dropna(subset=['Position']).copy()
            if not r_valid.empty:
                r_valid['Position'] = r_valid['Position'].astype(int)
                winner_row = r_valid[r_valid['Position'] == 1]
                winner = winner_row['Abbreviation'].iloc[0] if not winner_row.empty else None
                podium = r_valid[r_valid['Position'] <= 3].sort_values('Position')['Abbreviation'].tolist()
                # Extract ordered results for Elo updates
                race_results = r_valid.sort_values('Position')['Abbreviation'].tolist()
    except Exception:
        pass  # Race data unavailable

    return {
        'pole': pole,
        'winner': winner,
        'podium': podium,
        'quali_results': quali_results,  # Reusable for Elo updates
        'race_results': race_results,    # Reusable for Elo updates
    }

def brier_score(predictions: list[dict], actuals: list[str]) -> float:
    """Calculate Brier score for probabilistic predictions.

    Computes per-race Brier scores and averages them. Each race score is
    the mean squared error between predicted probabilities and outcomes.

    Note: Validates that probabilities are in [0, 1] range.
    """
    race_scores = []
    for pred, actual in zip(predictions, actuals):
        if actual is None or not pred:
            continue
        # Validate probabilities
        probs = list(pred.values())
        if not all(0 <= p <= 1 for p in probs):
            print(f"Warning: Invalid probabilities detected (not in [0,1])")
            continue
        # Calculate Brier score for this race
        race_score = 0.0
        for driver, prob in pred.items():
            outcome = 1.0 if driver == actual else 0.0
            race_score += (prob - outcome) ** 2
        # Average over number of drivers for this race
        race_scores.append(race_score / len(pred))
    return np.mean(race_scores) if race_scores else 1.0

def podium_accuracy(predictions: list[dict], actuals: list[dict]) -> float:
    """Calculate how often predicted top-3 matches actual podium."""
    correct = 0
    total = 0
    for pred, act in zip(predictions, actuals):
        if not act.get('podium'):
            continue
        # Get podium probabilities with safe access
        podium_probs = pred.get('podium_probabilities', {})
        if not podium_probs:
            continue
        # Get top 3 by probability
        predicted_podium = sorted(
            podium_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        predicted_drivers = {d for d, _ in predicted_podium}
        actual_podium = set(act['podium'])
        correct += len(predicted_drivers & actual_podium)
        total += 3
    return correct / total if total > 0 else 0.0

def calibration_analysis(predictions: list[dict], actuals: list[dict]) -> dict:
    """Analyze calibration of probability predictions."""
    all_probs = []
    all_outcomes = []

    for pred, act in zip(predictions, actuals):
        if not act.get('winner'):
            continue
        win_probs = pred.get('win_probabilities', {})
        if not win_probs:
            continue
        for driver, prob in win_probs.items():
            all_probs.append(prob)
            all_outcomes.append(1 if driver == act['winner'] else 0)

    if not all_probs:
        return {'prob_true': [], 'prob_pred': []}

    # Dynamically adjust bins based on sample size (min 10 samples per bin)
    n_bins = min(10, max(2, len(all_probs) // 10))
    try:
        prob_true, prob_pred = calibration_curve(all_outcomes, all_probs, n_bins=n_bins)
        return {'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()}
    except ValueError:
        # Not enough samples for calibration curve
        return {'prob_true': [], 'prob_pred': []}

def backtest_model(predictor_class, seasons: list[int], seed: int = 42) -> dict:
    """Backtest predictions against historical results using Brier scores.

    Uses a fresh predictor for each race to avoid data leakage. Only historical
    races (before the one being predicted) are used for Elo updates.

    Args:
        predictor_class: Class to instantiate for predictions
        seasons: List of seasons to backtest
        seed: Random seed for reproducibility (default: 42)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    predictions, actuals = [], []
    for season in seasons:
        races = get_races(season)
        # Create fresh predictor for each season to avoid leakage
        predictor = predictor_class()
        for race in races:
            try:
                # Update Elo only with races BEFORE current one (no future data)
                # This is handled by predict_weekend's incremental loading
                pred = predictor.predict_weekend(season, race)
                act = get_actual_results(season, race)
                predictions.append(pred)
                actuals.append(act)
                # Update Elo with actual results for next prediction
                # Use actual results directly instead of reloading sessions
                try:
                    # Update quali Elo from actual results (faster than reloading)
                    if act.get('quali_results'):
                        predictor.elo_system.update_quali_ratings(act['quali_results'])

                    # Update race Elo from actual results
                    if act.get('race_results'):
                        predictor.elo_system.update_race_ratings(act['race_results'])
                except Exception:
                    pass  # Skip Elo update if data unavailable
            except Exception as e:
                print(f"Skipping {season} {race}: {e}")
    return {
        'pole_brier': brier_score([p['pole_probabilities'] for p in predictions], [a['pole'] for a in actuals]),
        'win_brier': brier_score([p['win_probabilities'] for p in predictions], [a['winner'] for a in actuals]),
        'podium_accuracy': podium_accuracy(predictions, actuals),
        'calibration_curve': calibration_analysis(predictions, actuals),
        'n_races': len(predictions),
    }
```

## 10. Configuration (src/config.py)

```python
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
```

## 11. Notes

### Caching Strategy

FastF1 has built-in caching that stores session data locally:

```python
import fastf1
fastf1.Cache.enable_cache('./cache')  # Enable before loading any sessions
```

For API calls to Jolpica/OpenF1, implement request caching:

```python
from requests_cache import CachedSession

session = CachedSession(
    'f1_api_cache',
    expire_after=3600,  # 1 hour for live data
    allowable_codes=[200],
)
```

### Error Handling

Key scenarios to handle:

1. **Missing sessions**: FP1/FP2 may be cancelled (weather, accidents)
2. **Incomplete data**: Drivers may not set times in all sessions
3. **API failures**: Rate limiting, server errors
4. **New drivers**: Rookies without historical data need default ratings

```python
from src.config import DRIVER_TEAMS

class PredictionError(Exception):
    """Base exception for prediction errors."""
    pass

class InsufficientDataError(PredictionError):
    """Raised when not enough data to make prediction."""
    pass

def safe_predict(
    predictor,
    season: int,
    race: str,
    grid_penalties: dict | None = None,
    prediction_point: str = 'fp2',
    actual_grid: dict[str, int] | None = None,
):
    """Wrapper with graceful error handling.

    Args:
        predictor: F1Predictor instance
        season: Season year
        race: Race name
        grid_penalties: Optional grid penalties dict
        prediction_point: When predicting from ('fp1', 'fp2', 'fp3', 'quali', 'sprint')
        actual_grid: Optional actual grid positions (for post-quali predictions)

    Example usage:
        # Early prediction after FP2
        pred = safe_predict(predictor, 2025, 'Monaco', prediction_point='fp2')

        # Post-qualifying prediction with actual grid
        actual = {'VER': 1, 'NOR': 2, 'LEC': 3, ...}
        pred = safe_predict(predictor, 2025, 'Monaco', prediction_point='quali', actual_grid=actual)
    """
    try:
        return predictor.predict_weekend(
            season, race,
            grid_penalties=grid_penalties,
            prediction_point=prediction_point,
            actual_grid=actual_grid,
        )
    except InsufficientDataError as e:
        print(f"Warning: {e}, using fallback predictions")
        return generate_fallback_predictions(season, race, prediction_point)
    except Exception as e:
        print(f"Error predicting {race}: {e}")
        return None

def generate_fallback_predictions(season: int, race: str, prediction_point: str = 'fp2') -> dict:
    """Generate uniform fallback predictions when data is unavailable."""
    # Use previous season's grid or default driver list
    drivers = list(DRIVER_TEAMS.keys())
    n = len(drivers)
    if n == 0:
        # No drivers configured - return empty predictions
        return {
            'pole_probabilities': {},
            'win_probabilities': {},
            'podium_probabilities': {},
            'full_distributions': {},
            'weather': {'air_temp': 25, 'track_temp': 35, 'humidity': 50, 'rainfall': False, 'wind_speed': 5},
            'fallback': True,
            'prediction_point': prediction_point,
            'confidence': 'none',
            'grid_is_actual': False,
        }
    uniform_prob = 1.0 / n
    # Probability of being in top 3 = min(3/n, 1.0) to ensure valid probability
    podium_prob = min(3.0 / n, 1.0)

    return {
        'pole_probabilities': {d: uniform_prob for d in drivers},
        'win_probabilities': {d: uniform_prob for d in drivers},
        'podium_probabilities': {d: podium_prob for d in drivers},
        'full_distributions': {d: {i: uniform_prob for i in range(1, n + 1)} for d in drivers},
        'weather': {'air_temp': 25, 'track_temp': 35, 'humidity': 50, 'rainfall': False, 'wind_speed': 5},
        'fallback': True,
        'prediction_point': prediction_point,
        'confidence': 'none',  # Fallback has no confidence
        'grid_is_actual': False,
    }
```

### Handling Rookies

New drivers need initialization:

```python
import numpy as np
from src.config import DRIVER_TEAMS

def initialize_rookie(elo_system, driver: str, team: str):
    """Initialize rookie with team-based prior."""
    # Get average teammate rating as prior
    teammates = [d for d, t in DRIVER_TEAMS.items() if t == team and d != driver]
    if teammates:
        prior = np.mean([elo_system.ratings.get(t, {}).get('quali', 1500) for t in teammates])
    else:
        prior = 1400  # Conservative default for new teams

    elo_system.ratings[driver] = {
        'quali': prior - 50,  # Slight penalty for inexperience
        'race': prior - 50,
    }
```
