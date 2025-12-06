"""Feature engineering for F1 qualifying and race prediction."""

import fastf1
import pandas as pd
import numpy as np

from src.data_loader import F1DataLoader


class F1FeatureEngine:
    def __init__(self, data_loader: F1DataLoader):
        self.loader = data_loader
        self.laps: pd.DataFrame = pd.DataFrame()
        self.race_results: pd.DataFrame = pd.DataFrame()  # Grid/finish positions for overtake analysis

    def load_historical_data(self, seasons: list[int]):
        """Load and combine lap data from multiple seasons.

        Only loads Q and R sessions for historical data to minimize API calls.
        FP sessions are only loaded for current weekend predictions.
        """
        from datetime import datetime, timezone

        all_laps = []
        all_results = []
        today = datetime.now(timezone.utc)

        for season in seasons:
            try:
                schedule = fastf1.get_event_schedule(season)
            except Exception as e:
                print(f"Warning: Could not load {season} schedule for features: {e}")
                continue

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

                # Only load essential sessions for historical data (Q and R)
                # Skip FP1/FP2/FP3 to reduce API calls - they're only needed for current weekend
                for session_type in ['Q', 'R']:
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
        # Get last ~100 qualifying laps (5 recent races x ~20 laps per quali session)
        RECENT_LAPS = 5 * 20
        if self.laps.empty or 'Driver' not in self.laps.columns:
            recent = pd.DataFrame()
        else:
            recent = self.laps[
                (self.laps['Driver'] == driver) &
                (self.laps['session_type'] == 'Q')
            ].tail(RECENT_LAPS)

        return {
            'avg_gap_to_pole_pct': self._calc_gap_to_pole(recent),
            'teammate_delta': self._calc_teammate_delta(driver, recent),
            'circuit_affinity': self._calc_circuit_affinity(driver, circuit),
            'q3_conversion': self._calc_q3_rate(driver),
            'low_speed_strength': self._calc_sector_strength(driver, 'low_speed'),
            'high_speed_strength': self._calc_sector_strength(driver, 'high_speed'),
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
            # Team performance trend (upgrades/decline)
            'team_trend': self._calc_team_trend(driver),
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
        """Historical performance at this circuit and similar tracks vs average.

        Returns positive value if driver performs better at this circuit type,
        negative if worse. Range typically -0.5 to 0.5.

        Uses track type matching: Monaco performance also counts for Singapore, etc.
        """
        from src.config import get_similar_tracks, TRACK_TYPES

        if self.laps.empty:
            return 0.0

        driver_laps = self.laps[self.laps['Driver'] == driver]
        if driver_laps.empty:
            return 0.0

        # Get similar tracks for this circuit type
        similar_tracks = get_similar_tracks(circuit)
        target_circuits = [circuit] + similar_tracks

        # Get driver's laps at this circuit and similar ones
        circuit_laps = driver_laps[driver_laps['circuit'].isin(target_circuits)]
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

    def _calc_team_trend(self, driver: str, recent_races: int = 5) -> float:
        """Calculate team performance trend (are they improving or declining?).

        Compares team's average finish position in last N races vs earlier season.
        Returns positive if improving, negative if declining.
        Range: -0.5 (major decline) to 0.5 (major improvement like McLaren 2024).
        """
        if self.race_results.empty:
            return 0.0

        # Get driver's team
        driver_races = self.race_results[self.race_results['driver'] == driver]
        if driver_races.empty:
            return 0.0

        # Get all results for this driver's most recent season
        if 'season' not in driver_races.columns:
            return 0.0

        latest_season = driver_races['season'].max()
        season_results = driver_races[driver_races['season'] == latest_season].copy()

        if len(season_results) < recent_races + 2:
            return 0.0  # Not enough data

        # Sort by race order (circuit can approximate this)
        season_results = season_results.reset_index(drop=True)

        # Compare last N races vs earlier races
        recent = season_results.tail(recent_races)['finish'].dropna()
        earlier = season_results.head(len(season_results) - recent_races)['finish'].dropna()

        if recent.empty or earlier.empty:
            return 0.0

        recent_avg = recent.mean()
        earlier_avg = earlier.mean()

        # Positive = improving (lower finish position is better)
        # Normalize: 5 position improvement = 0.5 trend score
        improvement = (earlier_avg - recent_avg) / 10
        return max(-0.5, min(0.5, improvement))

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
