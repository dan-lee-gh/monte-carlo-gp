"""Backtest the F1 prediction model against historical results."""

import argparse
from src.predictor import F1Predictor
from src.validation import backtest_model


def main():
    parser = argparse.ArgumentParser(description='Backtest F1 Predictions')
    parser.add_argument('--seasons', type=int, nargs='+', default=[2024],
                        help='Seasons to backtest (e.g., --seasons 2023 2024)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Backtesting F1 Prediction Model")
    print(f"Seasons: {args.seasons}")
    print(f"{'='*60}\n")

    print("Running backtest (this may take several minutes)...")
    print("Loading historical data and simulating predictions...\n")

    results = backtest_model(F1Predictor, args.seasons, seed=args.seed)

    print(f"{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}\n")

    print(f"Races analyzed: {results['n_races']}")
    print()

    # Brier scores (lower is better, 0 = perfect, 1 = worst)
    print("BRIER SCORES (lower = better, 0 = perfect)")
    print("-" * 40)
    pole_brier = results['pole_brier']
    win_brier = results['win_brier']

    # Interpret Brier scores
    def interpret_brier(score):
        if score < 0.1:
            return "Excellent"
        elif score < 0.15:
            return "Good"
        elif score < 0.2:
            return "Fair"
        elif score < 0.25:
            return "Poor"
        else:
            return "Bad"

    print(f"  Pole position: {pole_brier:.4f} ({interpret_brier(pole_brier)})")
    print(f"  Race winner:   {win_brier:.4f} ({interpret_brier(win_brier)})")
    print()

    # Baseline comparison (random guess with 20 drivers = 0.05 probability each)
    # Brier score for random = sum((0.05 - 0)^2 * 19 + (0.05 - 1)^2 * 1) / 20 = 0.0475
    random_brier = 0.0475
    print(f"  (Random baseline: {random_brier:.4f})")
    print(f"  Pole improvement vs random: {((random_brier - pole_brier) / random_brier * 100):.1f}%")
    print(f"  Win improvement vs random:  {((random_brier - win_brier) / random_brier * 100):.1f}%")
    print()

    # Podium accuracy
    print("PODIUM ACCURACY")
    print("-" * 40)
    podium_acc = results['podium_accuracy']
    print(f"  Correct podium picks: {podium_acc:.1%}")
    print(f"  (Random baseline: ~15%)")
    print()

    # Calibration analysis
    calibration = results['calibration_curve']
    if calibration['prob_true'] and calibration['prob_pred']:
        print("CALIBRATION (predicted vs actual probability)")
        print("-" * 40)
        for pred, actual in zip(calibration['prob_pred'], calibration['prob_true']):
            bar_pred = '#' * int(pred * 50)
            bar_actual = '*' * int(actual * 50)
            print(f"  Pred {pred:.0%}: {bar_pred}")
            print(f"  True {actual:.0%}: {bar_actual}")
            print()
    else:
        print("(Not enough data for calibration analysis)")

    print(f"{'='*60}")
    print("Backtest complete!")
    print()
    print("Interpretation:")
    print("- Brier < 0.15: Model adds value over random guessing")
    print("- Podium > 33%: Model predicts podium better than chance")
    print("- Good calibration: Predicted % matches actual win rate")


if __name__ == "__main__":
    main()
