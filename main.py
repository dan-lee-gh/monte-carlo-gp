"""Run F1 race predictions."""

import argparse
from src.predictor import F1Predictor


def main():
    parser = argparse.ArgumentParser(description='F1 Race Prediction')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    parser.add_argument('--race', type=str, required=True, help='Race name (e.g., "Abu Dhabi")')
    parser.add_argument('--prediction-point', type=str, default='fp2',
                        choices=['fp1', 'fp2', 'fp3', 'quali', 'sprint'],
                        help='When to predict from (default: fp2)')
    parser.add_argument('--simulations', type=int, default=10000,
                        help='Number of Monte Carlo simulations (default: 10000)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"F1 Race Prediction: {args.season} {args.race}")
    print(f"Prediction point: {args.prediction_point}")
    print(f"{'='*60}\n")

    print("Loading data and running simulations...")
    predictor = F1Predictor()

    try:
        results = predictor.predict_weekend(
            season=args.season,
            race=args.race,
            prediction_point=args.prediction_point,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Display results
    print(f"\nWeather: {'Wet' if results['weather'].get('rainfall') else 'Dry'}")
    print(f"Track temp: {results['weather'].get('track_temp', 'N/A')}C")
    print(f"Confidence: {results['confidence']}\n")

    # Pole position predictions
    print("POLE POSITION PROBABILITIES")
    print("-" * 40)
    pole_sorted = sorted(results['pole_probabilities'].items(), key=lambda x: x[1], reverse=True)
    for i, (driver, prob) in enumerate(pole_sorted[:10], 1):
        bar = '#' * int(prob * 30)
        print(f"{i:2}. {driver:4} {prob:6.1%} {bar}")

    # Race winner predictions
    print("\nRACE WINNER PROBABILITIES")
    print("-" * 40)
    win_sorted = sorted(results['win_probabilities'].items(), key=lambda x: x[1], reverse=True)
    for i, (driver, prob) in enumerate(win_sorted[:10], 1):
        bar = '#' * int(prob * 30)
        print(f"{i:2}. {driver:4} {prob:6.1%} {bar}")

    # Podium predictions
    print("\nPODIUM PROBABILITIES")
    print("-" * 40)
    podium_sorted = sorted(results['podium_probabilities'].items(), key=lambda x: x[1], reverse=True)
    for i, (driver, prob) in enumerate(podium_sorted[:10], 1):
        bar = '#' * int(prob * 30)
        print(f"{i:2}. {driver:4} {prob:6.1%} {bar}")

    print(f"\n{'='*60}")
    print("Prediction complete!")


if __name__ == "__main__":
    main()
