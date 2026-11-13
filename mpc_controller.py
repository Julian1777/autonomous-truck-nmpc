# MPC controller simulates the next N steps at every single timestep. Based on the results it picks the best steering and acceleration plan.

import casadi as ca
from dynamics import r4k_step

# decision variables X and U, states, and inputs opti.variable

# parameters are fixed numbers we hand to the solver before it starts thinking opti.parameter

# constraints are the rules that the controller isnt allowed to break opti.subject_to

# cost function is the score of how good the planned trajectory is opti.minimize



class MPCController:
    def __init__(self, N, dt, l_truck, l_trailer):
        self.opti = ca.Opti()
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


    def set_cost(self):
        pass


    def solve(self, current_state, reference_state):
        pass