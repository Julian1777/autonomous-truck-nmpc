import numpy as np


class trafficVehicle:
    def __init__(self, vehicle_type, state, l, y_ref=0.0, v0=25.0):
        self.vehicle_type = vehicle_type
        self.state = state
        self.L = l

        # idm params
        self.v0 = v0 # desired speed
        self.T = 1.5 # safe time headway
        self.s0 = 2.0 # minimum gap
        self.a_max = 1.5 # maximum acceleration
        self.b = 2.0 # comfortable braking deceleration
        self.delta_exp = 4 # acceleration exponent

        # steering pi params
        self.kp_y = 0.08
        self.kp_i = 0.005
        self.kp_theta = 0.8
        self.integral_y_err = 0.0

        

    def dynamics(self, state, u, m=np):
        # 4 state variable dynamics model for simplicity, ignoring the trailer dynamics for traffic vehicles

        px, py, vx, theta = state

        delta, a = u

        dpx = vx * m.cos(theta)
        dpy = vx * m.sin(theta)
        dvx = a
        dtheta = (vx / self.L) * m.tan(delta)

        return np.array([dpx, dpy, dvx, dtheta])



    def r4k_step(self, state, control, dt):
        k1 = self.dynamics(state, control)
        k2 = self.dynamics(state + (dt / 2) * k1, control)
        k3 = self.dynamics(state + (dt / 2) * k2, control)
        k4 = self.dynamics(state + dt * k3, control)

        # simpsons rule weights (1,2,2,1)  divided by 6

        return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    # IDM Intelligent Driver model for the traffic vehicles

    def idm_acceleration(self, lead_vehicle=None):
        # computes smooth acceleration a, if no lead vehicle, vehicle simply goes v0

        px, _, vx, _ = self.state

        if lead_vehicle is None:
            interaction_term = 0.0
        else:
            lead_px, _, lead_vx, _ = lead_vehicle.state

            s = max(lead_px - px - 5.0, 0.1) # distance to lead vehicle, with a minimum of 0.1 to avoid division by zero

            delta_v = vx - lead_vx # pos if closing in on lead vehicle


            # Calculates desired safe following distance based on speed, time headway and relative velocity.
            s_star = self.s0 + max(0, vx * self.T + (vx * delta_v) / (2.0 * np.sqrt(self.a_max * self.b))) # desired minimum gap

            # compares the desired with the actual gap. If gap is large, term is small, if gap is small, term is large
            interaction_term = (s_star / s) ** 2

        # calculates acceleration based on IDM formula
        a = self.a_max * (1 - (vx / self.v0) ** self.delta_exp - interaction_term)
        return np.clip(a, -5.0, self.a_max) # clip acceleration to be within reasonable bounds

    def steering_control(self, y_ref):
        _, py, _, theta = self.state

        y_err = y_ref - py
        self.integral_y_err = np.clip(self.integral_y_err + y_err * dt, -5.0, 5.0)

        delta = self.kp_y * y_err + self.ki_y * self.integral_y_err - self.kp_theta * theta

        return np.clip(delta, -0.5, 0.5) # clip steering angle -0.5 to 0.5 rad

if __name__ == "__main__":
    # test traffic vehicle and r4k functions

    dt = 0.1
    l = 6.0
    state = np.array([0.0, 3.0, 15.0, 0.0])  # initial state [px, py, vx, theta1]
    control = np.array([0.03, 0.0])  # control inputs [delta, a] [steering angle, acceleration]

    car = trafficVehicle(vehicle_type="car", state=state, l=l)

    print("Initial state:", state)
    print(car.r4k_step(state, control, dt))


