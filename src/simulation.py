"""Monte Carlo race simulation for F1 predictions."""

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
            # Front row drivers lose less positions on average, backmarkers can gain more
            # Position 1-3: tend to hold, Position 10+: more variance
            position_factor = min(1.5, 0.5 + car.position * 0.1)  # Higher variance for backmarkers
            start_delta = np.random.normal(0, position_factor)
            # Front positions have less to gain, back positions have more opportunity
            if car.position <= 3:
                start_delta = min(start_delta, 1.0)  # Limit gains for front row
            lap_time = base_lap_time - start_delta * 0.5
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
                # Position in their "group" is based on their relative position
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
            driver_deg = tire_deg.get(car.driver, 0.0)
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
