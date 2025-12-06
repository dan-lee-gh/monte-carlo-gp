"""Elo rating system for F1 qualifying and race performance."""

import numpy as np


class F1EloSystem:
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        self.base_k = k_factor
        self.k = k_factor  # Current K-factor (can be adjusted for recency)
        self.initial = initial_rating
        self.ratings: dict = {}

    def set_recency_weight(self, years_ago: float, race_index: int = 0, total_races: int = 24):
        """Adjust K-factor based on how old the data is.

        Recent data gets higher K (more impact on ratings).
        - Current season (0 years): K = base_k * 1.5, plus race-within-season decay
        - 1 year ago: K = base_k * 1.0
        - 2 years ago: K = base_k * 0.7
        - 3+ years ago: K = base_k * 0.5

        Args:
            years_ago: How many years old is this season
            race_index: Which race in the season (0 = first race, 23 = last)
            total_races: Total races in the season (default 24)
        """
        if years_ago <= 0:
            # Current season: apply race-within-season decay
            # Last race gets 1.5x, first race gets ~0.75x
            # This makes Qatar (race 23) worth 2x Bahrain (race 1)
            race_weight = 0.75 + (0.75 * race_index / max(1, total_races - 1))
            self.k = self.base_k * race_weight
        elif years_ago <= 1:
            self.k = self.base_k * 1.0
        elif years_ago <= 2:
            self.k = self.base_k * 0.7
        else:
            self.k = self.base_k * 0.5  # Older data: half weight

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        # Clamp exponent to prevent overflow (diff of 4000 points = exponent of 10)
        exponent = max(-10, min(10, (rating_b - rating_a) / 400))
        return 1 / (1 + 10 ** exponent)

    def update_quali_ratings(self, quali_results: list[tuple[str, float]]):
        """Update qualifying Elo ratings based on lap times.

        Args:
            quali_results: List of (driver, best_lap_time) tuples
        """
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
        """Predict pole position probabilities based on Elo ratings.

        Returns dict of driver -> probability of pole.
        """
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

    def get_rating(self, driver: str, rating_type: str = 'quali') -> float:
        """Get a driver's current rating."""
        return self.ratings.get(driver, {}).get(rating_type, self.initial)
