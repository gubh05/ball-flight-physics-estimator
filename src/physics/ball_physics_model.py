import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BallParams:
    mass_kg: float = 0.43        # Standard football mass
    radius_m: float = 0.11       # Standard football radius
    drag_coeff: float = 0.47     # Sphere drag coefficient
    lift_coeff: float = 0.25     # Magnus lift coefficient
    air_density: float = 1.225   # kg/m³ at sea level
    gravity: float = 9.81        # m/s²


@dataclass
class TrajectoryResult:
    t: np.ndarray            # time array (s)
    x: np.ndarray            # horizontal position (m)
    y: np.ndarray            # vertical position (m)
    vx: np.ndarray           # horizontal velocity (m/s)
    vy: np.ndarray           # vertical velocity (m/s)
    ax: np.ndarray           # horizontal acceleration (m/s²) — aerodynamic only
    ay: np.ndarray           # vertical acceleration (m/s²) — aerodynamic only (no gravity)
    spin_rpm: float
    launch_speed_mps: float
    launch_angle_deg: float


class BallPhysicsModel:
    """Forward physics simulation of a football trajectory.

    Uses RK4 numerical integration with drag (air resistance) and Magnus
    (spin-induced lift) forces. Stops when the ball returns to ground (y < 0).
    """

    def __init__(self, params: BallParams = None):
        self.params = params or BallParams()
        p = self.params
        self._area = math.pi * p.radius_m ** 2          # cross-sectional area (m²)
        self._drag_const = 0.5 * p.air_density * self._area * p.drag_coeff
        self._lift_const = 0.5 * p.air_density * self._area * p.lift_coeff

    def _derivatives(self, state: np.ndarray, spin_rad_s: float) -> np.ndarray:
        """Compute [dx/dt, dy/dt, dvx/dt, dvy/dt] for the RK4 integrator.

        State vector: [x, y, vx, vy]
        """
        _, _, vx, vy = state
        p = self.params

        speed = math.sqrt(vx ** 2 + vy ** 2)

        # Drag force (opposes velocity)
        if speed > 1e-9:
            drag_ax = -self._drag_const * speed * vx / p.mass_kg
            drag_ay = -self._drag_const * speed * vy / p.mass_kg
        else:
            drag_ax = drag_ay = 0.0

        # Magnus force: F = lift_const * |v|² * (spin_unit × velocity_unit)
        # Spin axis is z (out of plane); cross product gives (-vy, vx) direction.
        # spin > 0 → backspin → upward lift; spin < 0 → topspin → downward force.
        # spin = 0 → no Magnus force.
        if speed > 1e-9 and abs(spin_rad_s) > 1e-9:
            spin_sign = math.copysign(1.0, spin_rad_s)
            magnus_ax = -self._lift_const * speed * vy * spin_sign / p.mass_kg
            magnus_ay =  self._lift_const * speed * vx * spin_sign / p.mass_kg
        else:
            magnus_ax = magnus_ay = 0.0

        # Total aerodynamic acceleration (gravity handled separately in integrator)
        ax_aero = drag_ax + magnus_ax
        ay_aero = drag_ay + magnus_ay - p.gravity  # include gravity for integration

        return np.array([vx, vy, ax_aero, ay_aero])

    def simulate(
        self,
        launch_speed_mps: float,
        launch_angle_deg: float,
        spin_rpm: float,
        dt: float = 0.001,
        sample_rate_hz: int = 100,
    ) -> TrajectoryResult:
        """Run RK4 integration from launch until ball hits ground.

        Parameters
        ----------
        launch_speed_mps : Initial speed in m/s  [5, 35]
        launch_angle_deg : Launch angle in degrees [5, 60]
        spin_rpm         : Spin rate in RPM (positive=backspin, negative=topspin)
        dt               : Integration timestep (s) — 1 ms gives good accuracy
        sample_rate_hz   : Downsample rate for output arrays

        Returns
        -------
        TrajectoryResult
        """
        spin_rad_s = spin_rpm * 2 * math.pi / 60.0
        angle_rad = math.radians(launch_angle_deg)
        vx0 = launch_speed_mps * math.cos(angle_rad)
        vy0 = launch_speed_mps * math.sin(angle_rad)

        state = np.array([0.0, 0.0, vx0, vy0])

        sample_interval = int(round(1.0 / (sample_rate_hz * dt)))
        sample_interval = max(1, sample_interval)

        t_list, x_list, y_list, vx_list, vy_list, ax_list, ay_list = [], [], [], [], [], [], []

        t = 0.0
        step = 0
        launched = False  # skip the very first point to detect landing correctly

        while True:
            x, y, vx, vy = state

            # Record at sample rate
            if step % sample_interval == 0:
                # Aerodynamic accelerations only (no gravity) — what IMU reads
                aero = self._derivatives(state, spin_rad_s)
                ax_imu = aero[2] + self.params.gravity   # remove gravity term added in _derivatives
                ay_imu = aero[3] + self.params.gravity

                t_list.append(t)
                x_list.append(x)
                y_list.append(y)
                vx_list.append(vx)
                vy_list.append(vy)
                ax_list.append(ax_imu)
                ay_list.append(ay_imu)

            # Stop after the ball has risen and then returns to y < 0
            if launched and y < 0.0:
                break

            if y > 0.01:
                launched = True

            # RK4 step
            k1 = self._derivatives(state, spin_rad_s)
            k2 = self._derivatives(state + 0.5 * dt * k1, spin_rad_s)
            k3 = self._derivatives(state + 0.5 * dt * k2, spin_rad_s)
            k4 = self._derivatives(state + dt * k3, spin_rad_s)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            step += 1

            # Safety: cap at 30 seconds
            if t > 30.0:
                logger.warning("Simulation exceeded 30s — stopping early.")
                break

        result = TrajectoryResult(
            t=np.array(t_list),
            x=np.array(x_list),
            y=np.array(y_list),
            vx=np.array(vx_list),
            vy=np.array(vy_list),
            ax=np.array(ax_list),
            ay=np.array(ay_list),
            spin_rpm=spin_rpm,
            launch_speed_mps=launch_speed_mps,
            launch_angle_deg=launch_angle_deg,
        )
        logger.debug(
            "simulate: speed=%.1f angle=%.1f spin=%.0f → %d samples, range=%.1fm",
            launch_speed_mps, launch_angle_deg, spin_rpm, len(result.t), self.range_m(result),
        )
        return result

    def time_of_flight(self, result: TrajectoryResult) -> float:
        """Total flight time in seconds."""
        return float(result.t[-1])

    def max_height(self, result: TrajectoryResult) -> float:
        """Maximum altitude reached (m)."""
        return float(result.y.max())

    def range_m(self, result: TrajectoryResult) -> float:
        """Horizontal range at landing (m)."""
        return float(result.x[-1])
