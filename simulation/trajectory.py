import math
import yaml


class Trajectory:
    def __init__(self, T, x0, y0):
        self.T = T
        self.current_time = 0
        self.x0 = x0
        self.y0 = y0

    def get_velocity(self, t):
        raise NotImplementedError("Subclasses should implement this!")

    def get_initial_position(self):
        return (self.x0, self.y0)

    def step(self, dt):
        velocity = self.get_velocity(self.current_time)
        self.current_time += dt
        return velocity

    @staticmethod
    def from_yaml(file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        trajectory_type = config['trajectory']['type']
        params = config['trajectory']['params']

        if trajectory_type == "FixedVelocityTrajectory":
            return FixedVelocityTrajectory(**params)
        elif trajectory_type == "ReturnTrajectory":
            return ReturnTrajectory(**params)
        elif trajectory_type == "PiecewiseLinearTrajectory":
            return PiecewiseLinearTrajectory(**params)
        elif trajectory_type == 'FourPointTrajectory':
            return FourPointTrajectory(**params)
        elif trajectory_type == 'ThreePointTrajectory':
            return ThreePointTrajectory(**params)
        elif trajectory_type == 'TwoPointTrajectory':
            return TwoPointTrajectory(**params)
        elif trajectory_type == "StaticTrajectory":
            return StaticTrajectory(**params)
        else:
            raise ValueError(f"Unsupported trajectory type: {trajectory_type}")

    @staticmethod
    def from_dict(trajectory_type, params):

        if trajectory_type == "FixedVelocityTrajectory":
            return FixedVelocityTrajectory(**params)
        elif trajectory_type == "ReturnTrajectory":
            return ReturnTrajectory(**params)
        elif trajectory_type == "PiecewiseLinearTrajectory":
            return PiecewiseLinearTrajectory(**params)
        elif trajectory_type == 'FourPointTrajectory':
            return FourPointTrajectory(**params)
        elif trajectory_type == 'ThreePointTrajectory':
            return ThreePointTrajectory(**params)
        elif trajectory_type == 'TwoPointTrajectory':
            return TwoPointTrajectory(**params)
        elif trajectory_type == "StaticTrajectory":
            return StaticTrajectory(**params)
        else:
            raise ValueError(f"Unsupported trajectory type: {trajectory_type}")


class FixedVelocityTrajectory(Trajectory):
    def __init__(self, T, x0, y0, vx, vy):
        super().__init__(T, x0, y0)
        self.vx = vx
        self.vy = vy

    def get_velocity(self, t):
        if 0 <= t <= self.T:
            return (self.vx, self.vy)
        else:
            return (0, 0)  # No velocity outside the defined time interval


class ReturnTrajectory(Trajectory):
    def __init__(self, T, x0, y0, x1, y1):
        super().__init__(T, x0, y0)
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def get_velocity(self, t):
        if 0 <= t <= self.T / 2:
            return (2 * (self.x1 - self.x0) / self.T, 2 * (self.y1 - self.y0) / self.T)
        elif self.T / 2 < t <= self.T:
            return (2 * (self.x0 - self.x1) / self.T, 2 * (self.y0 - self.y1) / self.T)
        else:
            return (0, 0)  # No velocity outside the defined time interval


class PiecewiseLinearTrajectory(Trajectory):
    def __init__(self, T, points):
        super().__init__(T, points[0][0], points[0][1])
        self.points = points
        self.intervals = T / (len(points) - 1)

    def get_velocity(self, t):
        segment = int(t / self.intervals)
        if segment < len(self.points) - 1:
            x0, y0 = self.points[segment]
            x1, y1 = self.points[segment + 1]
            vx = (x1 - x0) / self.intervals
            vy = (y1 - y0) / self.intervals
            return (vx, vy)
        else:
            return (0, 0)  # No velocity after the last point


class TwoPointTrajectory(Trajectory):
    def __init__(self, T, x0, y0, x1, y1):
        super().__init__(T, x0, y0)
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def get_velocity(self, t):
        if 0 <= t <= self.T:
            return (2 * (self.x1 - self.x0) / self.T, 2 * (self.y1 - self.y0) / self.T)
        else:
            return (0, 0)  # No velocity outside the defined time interval


class ThreePointTrajectory(Trajectory):
    def __init__(self, T, x0, y0, x1, y1, x2, y2):
        super().__init__(T, x0, y0)
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_velocity(self, t):
        if 0 <= t <= self.T / 2:
            return (2 * (self.x1 - self.x0) / self.T, 2 * (self.y1 - self.y0) / self.T)
        elif self.T / 2 < t <= self.T:
            return (2 * (self.x2 - self.x1) / self.T, 2 * (self.y2 - self.y1) / self.T)
        else:
            return (0, 0)  # No velocity outside the defined time interval


class FourPointTrajectory(Trajectory):
    def __init__(self, T, x0, y0, x1, y1, x2, y2, x3, y3):
        super().__init__(T, x0, y0)
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3

    def get_velocity(self, t):
        if 0 <= t <= self.T / 3:
            return (3 * (self.x1 - self.x0) / self.T, 3 * (self.y1 - self.y0) / self.T)
        elif self.T / 3 < t <= 2 * self.T / 3:
            return (3 * (self.x2 - self.x1) / self.T, 3 * (self.y2 - self.y1) / self.T)
        elif 2 * self.T / 3 < t <= self.T:
            return (3 * (self.x3 - self.x2) / self.T, 3 * (self.y3 - self.y2) / self.T)
        else:
            return (0, 0)  # No velocity outside the defined time interval


class StaticTrajectory(Trajectory):
    def __init__(self, T, x0=0, y0=0):
        super().__init__(T, x0, y0)

    def get_velocity(self, t):
        return (0, 0)


def press_trajectory_points(radius, penetration, angle, opening):
    points = [
        (radius * 1.1 * math.cos(angle + opening / 2), radius * 1.1 * math.sin(angle + opening / 2)),
        (radius * penetration * math.cos(angle), radius * penetration * math.sin(angle)),
        (radius * 1.1 * math.cos(angle - opening / 2), radius * 1.1 * math.sin(angle - opening / 2))
    ]
    return points


def feel_trajectory_points(radius, penetration, angle):
    points = [
        (radius * 1.1 * math.cos(angle), radius * 1.1 * math.sin(angle)),
        (radius * penetration * math.cos(angle + (math.pi - 2 * angle) / 3),
         radius * penetration * math.sin(angle + (math.pi - 2 * angle) / 3)),
        (radius * penetration * math.cos(angle + 2 * (math.pi - 2 * angle) / 3),
         radius * penetration * math.sin(angle + 2 * (math.pi - 2 * angle) / 3)),
        (radius * 1.1 * math.cos(math.pi - angle), radius * 1.1 * math.sin(math.pi - angle))
    ]
    return points


def poke_trajectory_points(radius, penetration, angle):
    points = [
        (radius * 1.1 * math.cos(angle), radius * 1.1 * math.sin(angle)),
        (radius * penetration * math.cos(angle), radius * penetration * math.sin(angle))
    ]
    return points


def long_press_trajectory_points(radius, penetration, angle, opening):
    points = [
        (radius * 1.1 * math.cos(angle + opening / 2), radius * 1.1 * math.sin(angle + opening / 2)),
        (radius * penetration * math.cos(angle + opening / 2), radius * penetration * math.sin(angle + opening / 2)),
        (radius * penetration * math.cos(angle - opening / 2), radius * penetration * math.sin(angle - opening / 2)),
        (radius * 1.1 * math.cos(angle - opening / 2), radius * 1.1 * math.sin(angle - opening / 2))
    ]
    return points
