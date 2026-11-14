# MPC controller simulates the next N steps at every single timestep. Based on the results it picks the best steering and acceleration plan.

import casadi as ca
from dynamics import r4k_step
import numpy as np
import matplotlib.pyplot as plt

# decision variables X and U, states, and inputs opti.variable

# parameters are fixed numbers we hand to the solver before it starts thinking opti.parameter

# constraints are the rules that the controller isnt allowed to break opti.subject_to

# cost function is the score of how good the planned trajectory is opti.minimize



class MPCController:
    def __init__(self, N, dt, l_truck, l_trailer):

        self.opti = ca.Opti()

        self.opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0})

        self.N = N
        self.dt = dt
        self.l_truck = l_truck
        self.l_trailer = l_trailer

        # define decision variables
        self.X = self.opti.variable(5, N + 1)  # 5 for the 5 state variables, N+1 since k=0 is also a timestamp and we want to go up to N which is the number of time steps we want to simulate
        self.U = self.opti.variable(2, N)  # 2 for the 2 control inputs, only N since it doesnt have control inputs for k=0

        # define parameters
        self.x0 = self.opti.parameter(5)  # initial state parameter
        self.x_ref = self.opti.parameter(5, N + 1)  # reference trajectory parameter

        # modular helper methods
        self.add_constraints()
        self.set_cost()
        self.opti.solver('ipopt')  # set the solver to IPOPT


    def add_constraints(self):
        # to anchor the start of the predicted trajectory to the current state
        self.opti.subject_to(self.X[:, 0] == self.x0) # [:, 0] : select all rows, 0 first column

        for k in range(self.N):
            next_state = r4k_step(self.X[:, k], self.U[:, k], self.dt, self.l_truck, self.l_trailer, ca)
            self.opti.subject_to(self.X[:, k + 1] == next_state)

        # restrict steering angle -0.5 to 0.5 rad
        self.opti.subject_to(self.opti.bounded(-0.5, self.U[0, :], 0.5)) # [0, :] : select first row, all columns

        # restricht acceleration -3 to 3 m/s^2
        self.opti.subject_to(self.opti.bounded(-3, self.U[1, :], 3)) # [1, :] : select second row, all columns

        theta1 = self.X[3, 1:]
        theta2 = self.X[4, 1:] # [4, 1:] 4 selects 5th row, 1: selects all columns from row 1

        # anti jackknife constraint restrcting the articulation angle abs(theta1 - theta2) between +- 0.4rad across all N steps
        self.opti.subject_to(self.opti.bounded(-0.4, theta1-theta2, 0.4))

        # road boundary constraint, constricting lateral position py inside realisitc highwaylanes, -6m to 6m across all N steps
        self.opti.subject_to(self.opti.bounded(-6, self.X[1, 1:], 6))

    def set_cost(self):
        # Q penalizes deviation from desired state, R penalizes magnitude of control inputs, N is the prediction horizon
        # Diagonal matrix is used since we care about each state variable independently, so for px the rest of the column is 0
        Q = ca.diag([0.0, 10.0, 5.0, 1.0, 1.0])  # penalize lateral position py and velocity vx more than the other state variables
        R = ca.diag([10.0, 1.0])  # penalize steering angle delta more than acceleration a
        total_cost = 0


        # accumulate the cost over every step k in the prediction horizon N
        for k in range(self.N):
            # state deviation
            state_error = self.X[:, k] - self.x_ref[:, k]
            # control input
            u_k = self.U[:, k]

            total_cost += state_error.T @ Q @ state_error + u_k.T @ R @ u_k
            # state_error.T @ Q @ state_error penalizes state errors
            # u_k.T @ R @ u_k penalizes control inputs

        # minimize the total cost
        self.opti.minimize(total_cost)


    def solve(self, current_state, reference_state):
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.x_ref, reference_state)

        # sol holds the solved traj over N future steps
        sol = self.opti.solve()

        sol_arr = sol.value(self.U)  # returns 2D numpy array of shape (2, N). row 0 is steering across all N, and row 1 is acceleration

        return sol_arr[:, 0]  # return the first column of the array, which is the first control input for the current timestep


if __name__ == "__main__":
    # TEST MPC

    state = np.array([0.0, -2.5, 10.0, 0.0, 0.0])  # initial state [px, py, vx, theta1, theta2]

    N = 10  # prediction horizon
    dt = 0.1
    l_truck = 5.4
    l_trailer = 10.8

    reference_state = np.tile(np.array([[0.0], [5.0], [20.0], [0.0], [0.0]]), (1, N + 1)) # this is the refrence traj we want, straight line 20 m/s, 0 lateral position, 0 heading angle, 0 trailer angle
    # np.tile repeats the array for N steps, so the refrence traj is the same for all N steps

    # create MPC controller
    mpc = MPCController(N=N, dt=dt, l_truck=l_truck, l_trailer=l_trailer)

    # store traj
    traj = [state.copy()]

    for i in range(20):
        control_input = mpc.solve(state, reference_state)
        state = r4k_step(state, control_input, dt, l_truck, l_trailer)
        
        traj.append(state.copy())

        print(f"Step {i}: Control Input: {control_input}, State: {state}")

    # convert traj to numpy array for plotting
    traj = np.array(traj)    

    # plot the trajectory

    x = traj[:, 0]
    y = traj[:, 1]

    plt.figure(figsize=(10, 6))

    #plot xy traj

    plt.plot(x, y, 'b-o', label='MPC Trajectory')

    plt.scatter(x[0], y[0], color='green', label='Start', s=100)  # Start point
    plt.scatter(x[-1], y[-1], color='red', label='End', s=100)  # End point

    plt.axhline(0, color='gray', linestyle='--', label='Reference Path')  # Reference path

    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('MPC Controlled Truck Trajectory')

    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    plt.show()