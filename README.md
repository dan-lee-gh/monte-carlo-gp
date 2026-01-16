# Monte Carlo GP - F1 Race Prediction System

A Monte Carlo simulation-based F1 race prediction system using Elo ratings and FastF1 data.

## Features

- **Elo Rating System**: Separate qualifying and race Elo ratings for each driver
- **Monte Carlo Simulation**: 10,000+ race simulations with tire degradation, pit stops, safety cars, DRS, and overtakes
- **Multiple Prediction Points**: Predict from FP1, FP2, FP3, qualifying, or sprint
- **Backtesting**: Validate model accuracy against historical results with Brier scores
- **Offline Mode**: Run predictions using only cached data (no API calls)
- **External Elo Integration**: Use pre-computed historical data to reduce API calls by ~90%

## Installation

```bash
# Clone the repository
git clone https://github.com/dan-lee-gh/monte-carlo-gp.git
cd monte-carlo-gp

# Install dependencies (using uv)
uv sync
```

## Usage

### Build External Elo Cache (Recommended First Step)

Build the Elo cache from external historical data. This fetches race results from GitHub (not FastF1), avoiding API rate limits:

```bash
# Build cache for seasons up to 2024
python main.py --build-cache --season 2025
```

### Make Predictions

```bash
# Predict a race
python main.py --race "Abu Dhabi" --season 2025

# Predict from a specific point in the weekend
python main.py --race "Monaco" --prediction-point fp3

# Use offline mode (no API calls)
python main.py --race "Bahrain" --offline
```

### Backtest the Model

```bash
# Backtest against 2024 season
python backtest.py --seasons 2024

# Backtest multiple seasons
python backtest.py --seasons 2023 2024

# Backtest in offline mode
python backtest.py --seasons 2024 --offline
```

## CLI Options

### main.py

| Option               | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `--race`             | Race name (e.g., "Abu Dhabi", "Monaco")                      |
| `--season`           | Season year (default: 2025)                                  |
| `--prediction-point` | When to predict: fp1, fp2, fp3, quali, sprint (default: fp2) |
| `--simulations`      | Number of Monte Carlo simulations (default: 10000)           |
| `--offline`          | Use only cached data (no API calls)                          |
| `--build-cache`      | Build external Elo cache from GitHub data                    |

### backtest.py

| Option      | Description                                       |
| ----------- | ------------------------------------------------- |
| `--seasons` | Seasons to backtest (e.g., --seasons 2023 2024)   |
| `--seed`    | Random seed for reproducibility (default: 42)     |
| `--offline` | Use only cached data (no API calls)               |

## Caching System

The system uses a multi-level caching strategy:

- **FastF1 Cache** (`./cache/`): Session data from FastF1 API
- **Elo Cache** (`./cache/elo_ratings.json`): Computed Elo ratings
- **External Elo Cache** (`./cache/external_elo.json`): Pre-computed from GitHub data

### Cache Priority

When loading Elo ratings:

1. Own cached Elo (most recent, includes current season)
2. External pre-computed Elo (historical baseline, no API calls)
3. Fresh computation (requires API calls)

## Project Structure

```text
monte-carlo-gp/
├── main.py              # CLI for predictions
├── backtest.py          # CLI for backtesting
├── src/
│   ├── config.py        # Configuration and constants
│   ├── data_loader.py   # FastF1 data loading
│   ├── elo.py           # Elo rating system
│   ├── external_elo.py  # External data integration
│   ├── features.py      # Feature engineering
│   ├── predictor.py     # Main prediction logic
│   ├── simulation.py    # Monte Carlo race simulation
│   └── validation.py    # Backtesting framework
└── cache/               # Cached data (gitignored)
```

## License

MIT
