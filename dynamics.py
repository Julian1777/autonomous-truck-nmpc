import numpy as np

def truck_ode(state, control, l_truck, l_trailer, m=np):

    px, py, vx, theta1, theta2 = state[0], state[1], state[2], state[3], state[4]

    delta, a = control[0], control[1]

    dpx = vx * m.cos(theta1)
    dpy = vx * m.sin(theta1)

    dvx = a

    dtheta1 = (vx / l_truck) * m.tan(delta)

    dtheta2 = (vx / l_trailer) * m.sin(theta1 - theta2)

    if m is np:
        return np.array([dpx, dpy, dvx, dtheta1, dtheta2])
    else:
        return m.vertcat(dpx, dpy, dvx, dtheta1, dtheta2)


# k1 estimate the intial slope, k2 estimate midpoint using k1, k3 improve midpoint estimate using k2, k4 estimate the endpoint using k3

def r4k_step(state, control, dt, l_truck, l_trailer, m=np):
    k1 = truck_ode(state, control, l_truck, l_trailer, m)
    k2 = truck_ode(state + (dt / 2) * k1, control, l_truck, l_trailer, m)
    k3 = truck_ode(state + (dt / 2) * k2, control, l_truck, l_trailer, m)
    k4 = truck_ode(state + dt * k3, control, l_truck, l_trailer, m)

    # simpsons rule weights (1,2,2,1)  divided by 6

    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


if __name__ == "__main__":
    # test truck_ode and r4k functions

    dt = 0.1
    l_truck = 5.4
    l_trailer = 10.8
    state = np.array([0.0, 0.0, 15.0, 0.0, 0.0])  # initial state [px, py, vx, theta1, theta2]
    control = np.array([0.03, 0.0])  # control inputs [delta, a] [steering angle, acceleration]

    print("Initial state:", state)
    print(r4k_step(state, control, dt, l_truck, l_trailer))
