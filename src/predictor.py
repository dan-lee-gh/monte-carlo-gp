"""Main F1 race predictor combining all components."""

import numpy as np
import pandas as pd
from src.data_loader import F1DataLoader
from src.features import F1FeatureEngine
from src.elo import F1EloSystem
from src.simulation import RaceSimulator, RaceConfig
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

        # Load historical data: previous 1 season + current season races before this one
        # (Reduced from 3 years to minimize API calls and processing time)
        # Current season data is loaded via load_season_data which filters to past events only
        # Process in chronological order with recency weighting (recent = more weight)
        historical_seasons = sorted([s for s in range(season - 1, season + 1) if s > 2017])
        for hist_season in historical_seasons:
            if hist_season not in self._processed_seasons:
                try:
                    years_ago = season - hist_season
                    historical = self.data_loader.load_season_data(hist_season)

                    # For current season, weight each race individually (later = more weight)
                    total_races = len(historical['qualifying'])
                    for race_idx, quali_result in enumerate(historical['qualifying']):
                        self.elo_system.set_recency_weight(years_ago, race_idx, total_races)
                        self.elo_system.update_quali_ratings(quali_result)

                    for race_idx, sq_result in enumerate(historical.get('sprint_qualifying', [])):
                        self.elo_system.set_recency_weight(years_ago, race_idx, total_races)
                        self.elo_system.update_quali_ratings(sq_result)

                    for race_idx, race_result in enumerate(historical['races']):
                        self.elo_system.set_recency_weight(years_ago, race_idx, total_races)
                        self.elo_system.update_race_ratings(race_result)

                    for race_idx, sprint_result in enumerate(historical.get('sprints', [])):
                        self.elo_system.set_recency_weight(years_ago, race_idx, total_races)
                        self.elo_system.update_race_ratings(sprint_result)

                    self._processed_seasons.add(hist_season)
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

        # Apply team trend adjustments (upgrades = faster pace)
        for driver in drivers:
            team_trend = race_features[driver].get('team_trend', 0)
            # Strong improving team (0.5) gains ~0.3s/lap, declining team loses pace
            base_pace[driver] = base_pace.get(driver, 90.0) - (team_trend * 0.6)

        # Apply wet weather specialist adjustments to pace
        if track_condition in ('damp', 'wet'):
            for driver in drivers:
                wet_skill = race_features[driver].get('wet_performance', 0)
                # Wet specialists gain up to 0.5s/lap advantage (add to existing pace)
                base_pace[driver] = base_pace[driver] - (wet_skill * 0.5)

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
        # Elo provides historical baseline, features adjust for current form
        elo_probs = self.elo_system.predict_quali_probs(drivers)

        # Boost/penalize based on teammate comparison (catches rookie potential, team changes)
        for driver in drivers:
            teammate_delta = features.get(driver, {}).get('teammate_delta', 0)
            if teammate_delta != 0 and driver in elo_probs:
                # If beating teammate by 0.2%, boost probability by ~5%
                teammate_boost = 1 + (teammate_delta * 0.25)
                elo_probs[driver] = elo_probs[driver] * max(0.5, min(1.5, teammate_boost))
        # Renormalize after teammate adjustments
        total = sum(elo_probs.values())
        if total > 0:
            elo_probs = {d: p / total for d, p in elo_probs.items()}

        # Return position probability distribution per driver
        # Position 0 = P1, Position 1 = P2, etc.
        n = len(drivers)
        result = {}
        for driver in drivers:
            base_prob = elo_probs.get(driver, 1 / n)
            # Adjust base probability using features (form score and circuit affinity)
            driver_features = features.get(driver, {})
            form_adj = driver_features.get('form_score', 0) * 0.15  # form contributes up to 15%
            circuit_adj = driver_features.get('circuit_affinity', 0) * 0.10  # circuit adds up to 10%
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
