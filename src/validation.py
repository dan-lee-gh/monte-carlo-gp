"""Validation framework for F1 predictions."""

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
