"""Data loading utilities for F1 session data via FastF1."""

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

        results = {'qualifying': [], 'races': [], 'sprints': [], 'sprint_qualifying': []}

        try:
            schedule = fastf1.get_event_schedule(season)
        except Exception as e:
            print(f"Warning: Could not load {season} schedule: {e}")
            return results

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
