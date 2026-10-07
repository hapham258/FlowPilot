import numpy as np
from flowpilot import FlowPilotNavigator, GpsGoal, Route, RouteGoal, compass_to_yaw

nav = FlowPilotNavigator(
    variant="flowpilot-dst-small", device="cpu"
)  # downloads the graph

obs = np.random.rand(1, nav.context_size, 3, 216, 384).astype(
    np.float32
)  # (B, T, 3, H, W) in [0, 1]
ego_vw = [1.2, 0.0]  # measured speed (m/s) and yaw rate (rad/s)

# Point goal: metres in the ego frame, x forward, y left
vw, plan = nav.inference_vw(obs, [8.0, 0.5], ego_vw)  # pure pursuit, vw: (B, 2)
vw, plan = nav.inference_vw(obs, [8.0, 0.5], ego_vw, controller="pd")  # PD
traj, scores = nav.inference_trajectory(
    obs, [8.0, 0.5], ego_vw
)  # (B, 6, 80, 5), (B, 6)

# GPS goal: a waypoint and the robot's fix, converted to a point goal
goal = GpsGoal(
    goal=(34.06901, -118.44512), robot=(34.06893, -118.44520), yaw=compass_to_yaw(45.0)
)
vw, plan = nav.inference_vw(obs, goal, ego_vw)

# Route goal: the route patch around the robot, plus the route point 12 m ahead as the point goal
route = Route(
    [[0.0, 0.0], [25.0, 0.0], [25.0, 40.0]], crosswalks=None
)  # world metres; Route.from_gps for lat / lon
vw, plan = nav.inference_vw(obs, RouteGoal(route, pose=(3.0, 0.2, 0.0)), ego_vw)

vw, plan = nav.step(
    obs[0, -1], [8.0, 0.5], ego_vw
)  # streaming: one frame per call at 20 Hz
nav.reset()  # between episodes
