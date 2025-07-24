import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from scipy.linalg import pinv
from copy import deepcopy
import model_functions as mf


class DualModelStructure:
    """
    For Game Theory Optimal controller
    """

    def __init__(
        self,
        h: float,
        N: int,
        x0,
        A,
        B1,
        B2,
        Q1,
        Q2,
        R11,
        R22,
        R12,
        R21,
        C1,
        C2,
        sensor_noise_arr1,
        sensor_noise_arr2,
        process_noise_value,
        internal_model_noise_value,
        W1_cov,
        V1_cov,
        W2_cov,
        V2_cov,
        sensory_delay,
    ):
        self.x0 = x0
        self.sensor_noise_arr1 = sensor_noise_arr1
        self.sensor_noise_arr2 = sensor_noise_arr2
        self.process_noise_value = process_noise_value
        self.internal_model_noise_value = internal_model_noise_value
        self.sensory_delay = sensory_delay

        self.A = A
        self.B1 = B1
        self.B2 = B2
        self.Q1 = Q1
        self.Q2 = Q2
        self.R11 = R11
        self.R22 = R22
        self.R12 = R12
        self.R21 = R21
        self.C1 = C1
        self.C2 = C2
        num_states = A.shape[0]  # number of states

        self.Ad = np.eye(num_states) + A * h
        # Augment A
        self.A_aug = mf.augment_A_matrix(self.A, sensory_delay)
        self.Ad_aug = mf.augment_A_matrix(self.Ad, sensory_delay)
        self.I = np.eye(self.Ad_aug.shape[0])  # Identity Matrix

        self.B1d = self.B1 * h
        self.B2d = self.B2 * h
        # AUgment B
        self.B1_aug = mf.augment_B_matrix(self.B1, sensory_delay)
        self.B2_aug = mf.augment_B_matrix(self.B2, sensory_delay)
        self.B1d_aug = mf.augment_B_matrix(self.B1d, sensory_delay)
        self.B2d_aug = mf.augment_B_matrix(self.B2d, sensory_delay)

        # Augment Q
        self.Q1N = Q1
        self.Q1 = Q1
        self.Q1_aug = mf.augment_Q_matrix(self.Q1, sensory_delay)
        self.Q1d = self.Q1 * h
        self.Q1Nd = self.Q1N * h

        self.Q2N = Q2
        self.Q2 = Q2
        self.Q2_aug = mf.augment_Q_matrix(self.Q2, sensory_delay)
        self.Q2d = self.Q2 * h
        self.Q2Nd = self.Q2N * h

        # Make Q zeros except for the last if we don't pass specific timesteps
        if self.Q1_aug.ndim < 3:
            temp = np.tile(
                np.zeros(self.Q1_aug.shape), (N, 1, 1)
            )  # Want a (1000,2,2) matrix for 1000 timesteps and only 2 states [[1,0],[0,1]]
            temp[-1, :, :] = self.Q1_aug
            self.Q1_aug = temp
        if self.Q2_aug.ndim < 3:
            temp = np.tile(
                np.zeros(self.Q2_aug.shape), (N, 1, 1)
            )  # Want a (1000,2,2) matrix for 1000 timesteps and only 2 states [[1,0],[0,1]]
            temp[-1, :, :] = self.Q2_aug
            self.Q2_aug = temp

        self.Q1d_aug = self.Q1_aug * h
        self.Q2d_aug = self.Q2_aug * h

        # Discretize R
        self.R11d = R11 * h
        self.R22d = R22 * h
        self.R12d = R12 * h
        self.R21d = R21 * h

        # Augment C matrix
        self.C1_aug = np.block([[np.zeros((C1.shape[0], C1.shape[1] * (sensory_delay))), C1]])
        self.C2_aug = np.block([[np.zeros((C2.shape[0], C2.shape[1] * (sensory_delay))), C2]])

        self.x0_aug = np.tile(x0, (sensory_delay + 1, 1))

        # Covariance matrices for KALMAN FILTER, not used in LQR
        self.W1 = W1_cov
        self.W2 = W2_cov

        # NOTE Process noise covariance can be different than the actual process noise.
        # This becomes particularly necessary for center cursor and target jumps. There's no actual noise on tose things (or the noise on their movement is watererd down)
        # like force -> vel -> pos. So the process covariance is kinda a fudge factor that allows updates to things that aren't explicitly modeled (like cursor and target jumps)
        self.V1 = V1_cov
        self.V2 = V2_cov

        # Want the process noise to be on the current time step, so all zeros after process noise
        self.V1_aug = mf.augment_Q_matrix(self.V1, sensory_delay=sensory_delay)
        self.V2_aug = mf.augment_Q_matrix(self.V2, sensory_delay=sensory_delay)

        # * Internal model covariance
        self.E1 = np.diag(np.eye(self.A.shape[0]) * internal_model_noise_value**2)
        self.E2 = np.diag(np.eye(self.A.shape[0]) * internal_model_noise_value**2)


def find_coupled_feedback_gains_continuous(
    A,
    B1,
    B2,
    Q1N,
    Q2N,
    R11,
    R22,
    R12,
    R21,
    hit_time,
    hold_steps,
    h,
    N,
    partner_knowledge,
):
    # ! WARNING: This works ok, but there's weird behaviour at the end of the simulation.. somehow I'm off by a timestep or something...Maybe from using the
    # ! the continuous solution of P then discrete for the feedback gain F?
    def dPdt(t, P, A, B1, B2, Q1, Q2, R11, R22, R12, R21, hit_time, partner_knowledge=True):
        """
        From Papavassilopoulos et al. (1979)
        "On the existence of Nash strategies and solutions to coupled riccati equations in linear-quadratic games"

        and Engwerda LQ Dynamic Optimization book
        """
        P1 = P[: A.shape[0] ** 2]  # First half of flattened initial solution is P1
        P2 = P[A.shape[0] ** 2 :]  # Second half of flattened initial solution is p2
        P1m = P1.reshape(A.shape[0], A.shape[0])  # Reshape into matrix form
        P2m = P2.reshape(A.shape[0], A.shape[0])

        #! No using hold time in the sims, they work fine without it
        # If there is a cost through all the time steps, I think I get the y-position feature as well
        if t > hit_time:
            Q1k = Q1
            Q2k = Q2
        else:
            Q1k = np.zeros_like(Q1)
            Q2k = np.zeros_like(Q2)

        if not partner_knowledge:
            P2m = np.zeros_like(P2m)

        # Can't have time varying Q in this yet, would need a workaround for time varying parameter
        # assert isinstance(Q1,(int,float))

        P1sol = -(
            P1m @ A
            + A.T @ P1m
            + Q1k
            - P1m @ B1 @ pinv(R11) @ B1.T @ P1m
            - P1m @ B2 @ pinv(R22) @ B2.T @ P2m
            - P2m @ B2 @ pinv(R22) @ B2.T @ P1m
            + P2m @ B2 @ pinv(R22) @ R12 @ pinv(R22) @ B2.T @ P2m
        ).flatten()
        P2sol = -(
            P2m @ A
            + A.T @ P2m
            + Q2k
            - P2m @ B2 @ pinv(R22) @ B2.T @ P2m
            - P2m @ B1 @ pinv(R11) @ B1.T @ P1m
            - P1m @ B1 @ pinv(R11) @ B1.T @ P2m
            + P1m @ B1 @ pinv(R11) @ R21 @ pinv(R11) @ B1.T @ P1m
        ).flatten()
        sol = np.hstack((P1sol, P2sol))
        return sol

    # * This reflects knowledge of the other players Q, not necessarily caring about it
    # * hstacked bc it's split up into P1 and P2 in the function
    Q1_init = np.hstack((Q1N.flatten(), Q2N.flatten()))
    Q2_init = np.hstack(
        (Q2N.flatten(), Q1N.flatten())
    )  # Swapped because P2 is the first solution, coupled with predictions

    P1_solution = solve_ivp(
        dPdt,
        t_span=(hit_time + hold_steps, 0),
        y0=Q1_init,  # First P values
        args=(A, B1, B2, Q1N, Q2N, R11, R22, R12, R21, hit_time, partner_knowledge),
        t_eval=np.arange(hit_time + hold_steps, -h, -h).round(3),
        method="RK45",
        max_step=h,
    )
    if P1_solution.status == -1:
        print(P1_solution)
        raise (ValueError("solution failed"))

    P2_solution = solve_ivp(
        dPdt,
        t_span=(hit_time + hold_steps, 0),
        y0=Q2_init,
        args=(
            A,
            B2,
            B1,
            Q2N,
            Q1N,
            R22,
            R11,
            R21,
            R12,
            hit_time,
            partner_knowledge,
        ),  #! NEED TO SWAP EVERY ARG OF 1 and 2 from above!
        t_eval=np.arange(hit_time + hold_steps, -h, -h).round(3),
        method="RK45",
        max_step=h,
    )
    if P2_solution.status == -1:
        print(P2_solution)
        raise (ValueError("solution failed"))
    P1_sol = np.array(
        [
            P1_solution.y[: A.shape[0] ** 2, i].reshape(A.shape[0], A.shape[0])
            for i in reversed(range(len(P1_solution.t)))
        ]
    )
    P1_predict_P2_sol = np.array(
        [
            P1_solution.y[A.shape[0] ** 2 :, i].reshape(A.shape[0], A.shape[0])
            for i in reversed(range(len(P1_solution.t)))
        ]
    )
    P2_sol = np.array(
        [
            P2_solution.y[: A.shape[0] ** 2, i].reshape(A.shape[0], A.shape[0])
            for i in reversed(range(len(P2_solution.t)))
        ]
    )
    P2_predict_P1_sol = np.array(
        [
            P2_solution.y[A.shape[0] ** 2 :, i].reshape(A.shape[0], A.shape[0])
            for i in reversed(range(len(P2_solution.t)))
        ]
    )

    assert np.all(P1_sol[-1] == Q1N)
    assert np.all(P2_sol[-1] == Q2N)
    Ad = np.eye(A.shape[0]) + A * h
    B1d = B1 * h
    B2d = B2 * h
    R11d = R11 * h
    R22d = R22 * h

    # F1 = np.zeros((N,B1d.shape[1],A.shape[0]))*np.nan
    # F2 = np.zeros((N,B1d.shape[1],A.shape[0]))*np.nan
    # for i in range(N):
    #     F1[i] = np.linalg.pinv(R11d + B1d.T @ P1_sol[i+1] @ B1d) @ (B1d.T @ P1_sol[i+1] @ Ad) # Feedback gain for current timestep with current Pk
    #     F2[i] = np.linalg.pinv(R22d + B2d.T @ P2_sol[i+1] @ B2d) @ (B2d.T @ P2_sol[i+1] @ Ad) # Feedback gain for current timestep with current Pk

    # Vectorized form of the above commented out loop, discrete solution
    F1 = np.linalg.pinv(R11d + B1d.T @ P1_sol[1:] @ B1d) @ (B1d.T @ P1_sol[1:] @ Ad)
    F2 = np.linalg.pinv(R22d + B2d.T @ P2_sol[1:] @ B2d) @ (B2d.T @ P2_sol[1:] @ Ad)
    # assert np.all(F1_test == F1)

    F1_predict_F2 = np.linalg.pinv(R22d + B2d.T @ P1_predict_P2_sol[1:] @ B2d) @ (B2d.T @ P1_predict_P2_sol[1:] @ Ad)
    F2_predict_F1 = np.linalg.pinv(R11d + B1d.T @ P2_predict_P1_sol[1:] @ B1d) @ (B1d.T @ P2_predict_P1_sol[1:] @ Ad)
    # F1_predict_F2 = (pinv(R22)@B1.T@P1_predict_P2_sol)
    # F2_predict_F1 = (pinv(R11)@B1.T@P2_predict_P1_sol)

    return P1_sol, P1_predict_P2_sol, P2_sol, P2_predict_P1_sol, F1, F1_predict_F2, F2, F2_predict_F1


def find_coupled_feedback_gains_discrete(
    A,
    B1,
    B2,
    Q1,
    Q2,
    R11,
    R22,
    R12,
    R21,
    N,
    partner_knowledge,
):
    S = np.zeros((N, *A.shape))  # storing A - sum(B@P)
    P1 = np.zeros((N + 1, *A.shape))
    P2 = np.zeros((N + 1, *A.shape))
    F1 = np.zeros((N, B1.shape[1], B1.shape[0]))
    F2 = np.zeros((N, B2.shape[1], B2.shape[0]))

    P1[-1] = Q1[-1]
    if partner_knowledge:  # Keeps P2 as zeros, assuming partner applies no control
        P2[-1] = Q2[-1]

    for i in reversed(range(N)):
        # Normal feedback gain part
        F1[i] = pinv(R11 + B1.T @ P1[i + 1] @ B1) @ (
            B1.T @ P1[i + 1] @ A - B1.T @ P1[i + 1] @ B2 @ F2[i]
        )  # note similarity for solo feedback gains, only difference is B1.T@P1@B2@F2
        F2[i] = pinv(R22 + B2.T @ P2[i + 1] @ B2) @ (B2.T @ P2[i + 1] @ A - B2.T @ P2[i + 1] @ B1 @ F1[i])

        # Ricatti solution
        S[i] = A - B1 @ F1[i] - B2 @ F2[i]
        P1[i] = S[i].T @ P1[i + 1] @ S[i] + F1[i].T @ R11 @ F1[i] + F2[i].T @ R12 @ F2[i] + Q1[i]
        if partner_knowledge:  # Keeps P2 as zeroes if no partner knowledge
            P2[i] = S[i].T @ P2[i + 1] @ S[i] + F2[i].T @ R22 @ F2[i] + F1[i].T @ R21 @ F1[i] + Q2[i]

    return P1, P2, F1, F2


def find_kalman_gain(Ad, C, W, V, N):
    """
    Ad: nx X nx
     - State transition matrix
    C: nz X nx
     - Observation Matrix
    W: nz X nz
     - Sensor/measurement covariance
    V: nx X nx
     - Process covariance
    N:
     - number of timesteps
    """
    ## Calculate the optimal Kalman gain (forward in time) ##
    # Intialize
    nx = Ad.shape[0]  # Number of states
    nz = C.shape[0]  # Number of sensory states
    P_prior = np.zeros((N + 1, nx, nx))  # Prior covariance !! CANNOT BE *np.nan, matrix multiplication gets weird
    G = np.zeros((N + 1, nz, nz)) * np.nan  # State innnovation
    K = np.zeros((N + 1, nx, nz)) * np.nan  # Kalman gain
    I = np.eye(K[0].shape[0])  # identity matrix
    P_post = np.zeros((N + 1, nx, nx)) * np.nan
    P_prior[0, -nz:, -nz:] = (
        W  # Initial process covariance is the W value, this makes it so we prefer info from the first measurements
    )

    for i in range(N + 1):
        G[i, :, :] = (
            C @ P_prior[i, :, :] @ C.T + W
        )  # Covariance based on observation matrix and prior covariance + measurement covariance
        K[i, :, :] = P_prior[i, :, :] @ C.T @ np.linalg.inv(G[i, :, :])  # Optimal Kalman gain for current timestep
        # P_post[i,:,:] = (I - K[i,:,:] @ C) @ P_prior[i,:,:] # Updated covariance estimation, unstable version don't think it's a problem tho
        P_post[i, :, :] = (I - K[i, :, :] @ C) @ P_prior[i, :, :] @ (I - K[i, :, :] @ C).T + K[i] @ W @ K[
            i
        ].T  # Updated covariance estimation, numerically stable version
        if i != N:
            P_prior[i + 1, :, :] = (
                Ad @ P_post[i, :, :] @ Ad.T + V
            )  # Next prior covariance estimation is the current posterior

    return K, P_post, P_prior


def find_dual_posterior_estimate(
    measurement,
    prior,
    u_self,
    u_partner,
    sensor_noise_arr,
    internal_model_noise,
    K,
    A,
    B_self,
    B_partner,
    C,
    nx,
    partner_knowledge,
    LQR,
):
    def _augment_state(x_pred, prev_x_aug, num_states):
        """
        Here we want to keep the prior states (previous posteriors), but then create the augmented prediction that uses
        those previous posterior states, and then tack on the new x_pred to the beginning

        That way, the observation matrix (which is [0,0,0,1]; crudely) only uses the LAST index of the states,
        which is the delayed posterior estimates
        """
        x_aug = deepcopy(prev_x_aug)  # Copy previous augmented state
        x_aug[num_states:] = prev_x_aug[
            :-num_states
        ]  # last part of x_aug is equal to first part of prev_x_aug (shifting down)
        x_aug[:num_states] = x_pred  # Most up to date x is beginning of indices (meaning that C doesn't observe it)
        return x_aug

    # Set u2 to 0's if no partner knowledge
    if not partner_knowledge:
        u_partner = np.zeros_like(u_self)

    # Observe delayed measurement according to C matrix, this goes to a nz x 1 (aka not augmented)
    sensor_noise = (
        np.zeros_like(sensor_noise_arr)[:, np.newaxis] if LQR else np.random.normal(0, sensor_noise_arr)[:, np.newaxis]
    )

    y_obs = C @ measurement + sensor_noise

    # Next prior state is current post, aka what I expect the next state to be based on current posterior state estimate
    prediction_noise = (
        np.zeros((B_self @ u_self).shape) if LQR else np.random.normal(0, internal_model_noise, (B_self @ u_self).shape)
    )
    prediction_noise[nx:] = 0  # just noise on current prediction
    x_prediction = A @ prior + B_self @ u_self + B_partner @ u_partner + prediction_noise
    x_pred_aug = _augment_state(
        x_pred=x_prediction[:nx],
        prev_x_aug=prior,  # prior comes from posterior estimate in run_dual_simulation
        num_states=nx,
    )  # Using the current x_prediction, but C makes it so we can't observe this yet
    x_post_next = x_pred_aug + K @ (
        y_obs - C @ x_pred_aug
    )  # Updated posterior estimate combines the prior and the kalman gain operating on the residual (y-C@prior)
    return y_obs, x_pred_aug, x_post_next


def run_dual_simulation(
    h,
    N,
    x0,
    Ad,
    B1d,
    B2d,
    C1,
    C2,
    K1,
    K2,
    F1,
    F2,
    nx,
    nz,
    sensor_noise_arr1,
    sensor_noise_arr2,
    process_noise_value,
    internal_model_noise,
    perturbation: list[float],
    probe_trial: bool,
    probe_duration: int,
    state_mapping,
    perturbation_states: list[str],
    partner_knowledge: bool,
    LQR: bool,
    fix_u1_val,
    fix_u2_val,
):
    # * Define empty arrays
    if True:
        x = np.zeros((N + 1, Ad.shape[0], 1)) * np.nan  # True states
        x1_prior = np.zeros((N + 1, Ad.shape[0], 1)) * np.nan
        x1_post = np.zeros((N + 1, Ad.shape[0], 1)) * np.nan
        y1_obs = np.zeros((N + 1, C1.shape[0], 1)) * np.nan  # Observed state with noise
        u1 = np.ones((N, B1d.shape[1], 1)) * fix_u1_val
        x2_prior = np.zeros((N + 1, Ad.shape[0], 1)) * np.nan
        x2_post = np.zeros((N + 1, Ad.shape[0], 1)) * np.nan
        y2_obs = np.zeros((N + 1, C2.shape[0], 1)) * np.nan  # Observed state with noise
        u2 = np.ones((N, B2d.shape[1], 1)) * fix_u2_val

        x_applied_force = np.zeros((N + 1, Ad.shape[0], 1)) * np.nan  # True states
        applied_force1 = np.zeros(N)
        applied_force2 = np.zeros(N)

    if probe_trial:
        B1d_state = deepcopy(B1d)
        B2d_state = deepcopy(B2d)
        B1d_state[state_mapping["rfx"], 0] = 0  # Turn off ability to move in the x-dimension
        B2d_state[state_mapping["lfx"], 0] = 0  # Turn off ability to move in the x-dimension
    else:
        B1d_state = deepcopy(B1d)
        B2d_state = deepcopy(B2d)

    # Only want noise on the indices where we have control
    noise_mask1 = np.max(B1d != 0, axis=1)  # Take the True value across all columns for every row
    noise_mask2 = np.max(B2d != 0, axis=1)  # Take the True value across all columns for every row
    process_noise1 = np.zeros_like(x0)
    process_noise2 = np.zeros_like(x0)
    # * Initialize arrays
    x[0, :, :] = x0
    x_applied_force[0, :, :] = x0

    y1_obs[0, :, :] = C1 @ x0
    x1_post[0, :, :] = x0  # initial condition, can't use prior because it's the first one

    y2_obs[0, :, :] = C2 @ x0
    x2_post[0, :, :] = x0  # initial condition, can't use prior because it's the first one

    # * Set up perturbation variables
    jump_flag = True
    jump_back_flag = True
    jump_step = 10000  # Big number so doesn't trigger if statement until the jump is actually initiated
    jump = np.zeros_like(x)  # Want this to be time varying
    linear_jump_steps = int(25 * (0.001 / h))  # 25ms linear jump out, like experiment
    jump_ids = [state_mapping[key] for key in perturbation_states]
    try:
        target_ypos_idx = state_mapping["cty"]
    except KeyError:
        target_ypos_idx = state_mapping["rty"]

    for t in range(0, N):
        # * Get optimal control signal
        u1[t] = -F1[t] @ (x1_post[t])
        u2[t] = -F2[t] @ (x2_post[t])

        # * If we've crossed 25% of the y distance, then set the linear jump
        if x[t, state_mapping["ccy"]] >= 0.25 * x[t, target_ypos_idx] and jump_flag:
            for jump_idx, pert_distance in zip(jump_ids, perturbation):
                jump[t : t + linear_jump_steps, jump_idx, 0] = np.repeat(
                    pert_distance / linear_jump_steps, linear_jump_steps
                )
            jump_flag = False  # set to false so we don't jump twice
            jump_step = t
        elif probe_trial:
            # Jump back
            if t >= jump_step + probe_duration and jump_back_flag:
                for jump_idx, pert_distance in zip(jump_ids, perturbation):
                    jump[t : t + linear_jump_steps, jump_idx, 0] = np.repeat(
                        -pert_distance / linear_jump_steps, linear_jump_steps
                    )
                jump_back_flag = False

        # * Set process noise in x direction only if they aren't constrained to force channels
        if not LQR:
            if not probe_trial:
                process_noise1[state_mapping["rfx"]] = np.random.normal(0, process_noise_value)
                process_noise2[state_mapping["lfx"]] = np.random.normal(0, process_noise_value)
            process_noise1[state_mapping["rfy"]] = np.random.normal(0, process_noise_value)
            process_noise2[state_mapping["lfy"]] = np.random.normal(0, process_noise_value)

        # * Update the true state
        x[t + 1] = Ad @ x[t] + B1d_state @ u1[t] + B2d_state @ u2[t] + process_noise1 + process_noise2 + jump[t]

        # * Update the probe trial force states
        if probe_trial:
            # Note that i'm using the original B
            x_applied_force[t + 1] = Ad @ x[t] + B1d @ u1[t] + B2d @ u2[t] + process_noise1 + process_noise2 + jump[t]
            # * Get applied force against a "channel"
            try:
                applied_force1[t] = x_applied_force[t, state_mapping["rfx"]] + np.random.normal(
                    0, process_noise_value
                )  # re-add noise so curves are different
                applied_force2[t] = x_applied_force[t, state_mapping["lfx"]] + np.random.normal(0, process_noise_value)
            except KeyError:
                applied_force1[t] = x_applied_force[t, state_mapping["rhvx"]]
                applied_force2[t] = x_applied_force[t, state_mapping["lhvx"]]

        # Find posterior estimate of next cursor state. This handles the augmentation and returns only true states
        y1_obs[t + 1], x1_prior[t + 1], x1_post[t + 1] = find_dual_posterior_estimate(
            measurement=x[t + 1],  # This is augmented, then C will select the delayed states
            prior=x1_post[t],  # Prior is the last posterior (t-1)
            u_self=u1[t],  # Last control signal for our prediction
            u_partner=u2[t],  # Last control signal for our prediction
            sensor_noise_arr=sensor_noise_arr1,
            internal_model_noise=internal_model_noise,  # not process noise, noise on the prediction step
            K=K1[t + 1],  # Current feedback gain
            nx=nx,
            A=Ad,
            B_self=B1d_state,
            B_partner=B2d_state,  #! Using the B1d_state, becuase it's likely that people realize they are in a force channel and update accordingly
            C=C1,
            partner_knowledge=partner_knowledge,
            LQR=LQR,
        )
        y2_obs[t + 1], x2_prior[t + 1], x2_post[t + 1] = find_dual_posterior_estimate(
            measurement=x[t + 1],  # This is augmented, then C will select the delayed states
            prior=x2_post[t],  # Prior is the last posterior (t-1)
            u_self=u2[t],  # Last control signal for our prediction
            u_partner=u1[t],  # Last control signal for our prediction
            sensor_noise_arr=sensor_noise_arr2,
            internal_model_noise=internal_model_noise,  # not process noise, noise on the prediction step
            K=K2[t + 1],  # Current feedback gain
            nx=nx,
            A=Ad,
            B_self=B2d_state,
            B_partner=B1d_state,
            C=C2,
            partner_knowledge=partner_knowledge,
            LQR=LQR,
        )

    return (
        u1.squeeze(),
        u2.squeeze(),
        x.squeeze(),
        x1_post.squeeze(),
        x2_post.squeeze(),
        applied_force1,
        applied_force2,
        jump_step,
    )


class DualLQG:
    """
    This class reflects two separate LQG models that
    """

    def __init__(self, p1_model: SingleLQG, p2_model: SingleLQG):
        self.p1_model = p1_model
        self.p2_model = p2_model
        self.timesteps = self.p1_model.timesteps
        self.h = self.p1_model.h
        self.sensory_delay = self.p1_model.sensory_delay
        self.perturbation = self.p1_model.perturbation
        self.probe_trial = self.p1_model.probe_trial
        self.probe_duration = self.p1_model.probe_duration
        self.state_mapping = self.p1_model.state_mapping

        # Here we're using the individual Model Structures from P1 and P2 to remake the DualLQG Model structure
        self.MS = DualModelStructure(
            x0=self.p1_model.MS.x0,
            A=self.p1_model.MS.A,
            B1=self.p1_model.MS.B,
            B2=self.p2_model.MS.B,
            Q1=self.p1_model.MS.Q,
            Q2=self.p2_model.MS.Q,
            R11=self.p1_model.MS.R,
            R22=self.p2_model.MS.R,
            R12=None,
            R21=None,
            C1=self.p1_model.MS.C,
            C2=self.p2_model.MS.C,
            W_cov=self.p1_model.MS.W_cov,
            V_cov=self.p1_model.MS.V_cov,
            sensor_noise_arr=self.p1_model.MS.sensor_noise_arr,
            process_noise_value=self.p1_model.MS.process_noise_value,
            T=self.p1_model.MS.T,
            h=self.h,
            sensory_delay=self.sensory_delay,
        )
        self.state_timesteps = np.arange(0, self.MS.T + self.h, self.h)

        self.P1 = p1_model.P
        self.P2 = p2_model.P
        self.F1 = p1_model.F
        self.F2 = p2_model.F

        self.K1 = p1_model.K
        self.K2 = p2_model.K

    def run_simulation(self):
        """
        - Use this if only doing one hand
        """

        (
            self.u1,
            self.u2,
            self.x,
            self.x1_post,
            self.x2_post,
            self.applied_force1,
            self.applied_force2,
            self.jump_time,
        ) = run_dual_simulation(
            MS=self.MS,
            x0=self.MS.x0_aug,
            Ad=self.MS.Ad_aug,
            B1d=self.MS.B1d_aug,
            B2d=self.MS.B2d_aug,
            C1=self.MS.C1_aug,
            C2=self.MS.C2_aug,
            perturbation=self.perturbation,
            probe_trial=self.probe_trial,
            probe_duration=self.probe_duration,
            state_mapping=self.state_mapping,
            K1=self.K1,
            K2=self.K2,
            F1=self.F1,
            F2=self.F2,
            nx=self.MS.A.shape[0],
        )

    def calculate_cost(self):
        energy_cost1 = 0
        energy_cost2 = 0
        for i in range(self.N):
            energy_cost1 += self.u1[i].T @ self.MS.R11 @ self.u1[i]
            energy_cost2 += self.u2[i].T @ self.MS.R22 @ self.u2[i]
        self.p1_cost = self.x[-1].T @ self.MS.Q1_aug[-1] @ self.x[-1] + energy_cost1
        self.p2_cost = self.x[-1].T @ self.MS.Q2_aug[-1] @ self.x[-1] + energy_cost2
        self.joint_cost = self.p1_cost + self.p2_cost


class GameTheoryLQG:
    def __init__(
        self,
        A,
        B1,
        B2,
        Q1,
        Q2,
        R11,
        R22,
        R12,
        R21,
        C1,
        C2,
        sensor_noise_arr1,
        sensor_noise_arr2,
        process_noise_value,
        internal_model_noise_value,
        W1_cov,
        W2_cov,
        V1_cov,
        V2_cov,
        x0,
        perturbation,
        sensory_delay,
        state_mapping,
        h=0.01,
        HIT_TIME=0.8,
        HOLD_TIME=0,
        name=None,
        p1_target=None,
        p2_target=None,
        partner_knowledge=True,
        probe_trial=False,
        probe_duration=0,
        alpha=0,
        perturbation_states=["ccx"],
        LQR=False,
        consider_partner_Q=True,
        fix_u1_val=np.nan,
        fix_u2_val=np.nan,
    ):
        """
        This class builds a single LQG. Depending on the B matrix it can be either a single LQG
        or an LQG with two hands (bimanual version)

        Kwargs
        A
        B
        Q
        R
        C
        W
        V
        x0: initial states
        T: final timepoint
        h
        """
        self.name = name
        self.p1_target = p1_target
        self.p2_target = p2_target
        self.sensory_delay = sensory_delay
        self.probe_trial = probe_trial
        self.probe_duration = probe_duration
        self.state_mapping = state_mapping
        self.h = h
        self.HIT_TIME = HIT_TIME
        self.HIT_STEP = int(
            self.HIT_TIME / self.h
        )  # Hit time is in seconds (i.e. 0.8s), so just divide by h to get the step
        self.HOLD_TIME = HOLD_TIME
        self.HOLD_STEP = int(
            self.HOLD_TIME / self.h
        )  # Hold time is in seconds (i.e. 0.5s), so just divide by h to get the step
        self.partner_knowledge = partner_knowledge
        self.alpha = alpha if partner_knowledge else 0
        self.LQR = LQR
        self.perturbation = [perturbation] if isinstance(perturbation, float) else perturbation
        self.perturbation_states = (
            [perturbation_states] if isinstance(perturbation_states, str) else perturbation_states
        )
        self.sensor_noise_arr1 = sensor_noise_arr1
        self.sensor_noise_arr2 = sensor_noise_arr2
        self.process_noise_value = process_noise_value

        self.timesteps = np.arange(0, self.HIT_TIME + self.HOLD_TIME, h).round(
            10
        )  # rounding bc np does weird things at 0, and scipy solve_ivp will screw up teval
        self.N = len(self.timesteps)
        self.backwards_timesteps = np.arange(self.HIT_TIME + self.HOLD_TIME - h, -h, -h).round(10)
        self.state_timesteps = np.arange(0, self.HIT_TIME + self.HOLD_TIME + h, h)  # Extra step for the states

        # So I can run simple perturbation experiments with other u2 values fixed
        self.fix_u1_val = fix_u1_val
        self.fix_u2_val = fix_u2_val

        if not partner_knowledge:
            R12 = R12 * 0
            R21 = R21 * 0

        if B1.shape[1] == 2:
            self.force_labels = ["fx", "fy"]
        elif B1.shape[1] == 4:
            self.force_labels = ["frx", "fry", "flx", "fly"]

        if self.probe_trial and self.probe_duration == 0:
            raise ValueError("probe duration is 0 on probe trial, make it a positive number")

        # self.Q1_cost = augment_Q_matrix(deepcopy(Q1),self.sensory_delay)
        # self.Q2_cost = augment_Q_matrix(deepcopy(Q2),self.sensory_delay)

        # * Assertion check to make sure they will hit the arget
        if False:
            assert np.all(
                (
                    Q1[..., self.state_mapping["ccx"], self.state_mapping["ccx"]]
                    + Q1[..., self.state_mapping["ccx"], self.state_mapping["rtx"]]
                    + Q2[..., self.state_mapping["ccx"], self.state_mapping["rtx"]]
                )
                == 0
            )

            assert np.all(
                (
                    Q1[..., self.state_mapping["ccx"], self.state_mapping["ccx"]]
                    + Q1[..., self.state_mapping["rtx"], self.state_mapping["ccx"]]
                    + Q2[..., self.state_mapping["rtx"], self.state_mapping["ccx"]]
                )
                == 0
            )
            assert np.all(
                (
                    Q2[..., self.state_mapping["ccx"], self.state_mapping["ccx"]]
                    + Q2[..., self.state_mapping["ccx"], self.state_mapping["ltx"]]
                    + Q1[..., self.state_mapping["ccx"], self.state_mapping["ltx"]]
                )
                == 0
            )

            assert np.all(
                (
                    Q2[..., self.state_mapping["ccx"], self.state_mapping["ccx"]]
                    + Q2[..., self.state_mapping["ltx"], self.state_mapping["ccx"]]
                    + Q1[..., self.state_mapping["ltx"], self.state_mapping["ccx"]]
                )
                == 0
            )

        if consider_partner_Q:
            if "rtx" in state_mapping and "ltx" in state_mapping:
                assert alpha <= 0.5
                # self.Q1 = self.care_about_partner_Q_two_targets(self_Q=Q1, partner_Q=Q2, alpha=self.alpha, player=1)
                # self.Q2 = self.care_about_partner_Q_two_targets(self_Q=Q2, partner_Q=Q1, alpha=self.alpha, player=2)
                self.Q1, self.Q2 = self.care_about_partner_Q_two_targets_v2(Q1=Q1, Q2=Q2, alpha1=alpha, alpha2=alpha)
            else:
                assert "ctx" in state_mapping
                self.Q1 = self.care_about_partner_Q(self_Q=Q1, partner_Q=Q2, alpha=self.alpha, player=1)
                self.Q2 = self.care_about_partner_Q(self_Q=Q2, partner_Q=Q1, alpha=self.alpha, player=2)

        else:
            self.Q1 = Q1
            self.Q2 = Q2

        # assert np.all(np.sum(self.Q1[state_mapping['ccx']:,:],axis=1) == 0)
        # assert np.all(np.sum(self.Q2[state_mapping['ccx']:,:],axis=1) == 0)
        assert sensory_delay < 80
        # Here we're using the individual Model Structures from P1 and P2 to remake the DualLQG Model structure
        self.MS = DualModelStructure(
            x0=x0,
            A=A,
            B1=B1,
            B2=B2,
            Q1=self.Q1,
            Q2=self.Q2,
            R11=R11,
            R22=R22,
            R12=R12,
            R21=R21,
            C1=C1,
            C2=C2,
            N=self.N,
            sensor_noise_arr1=sensor_noise_arr1,
            sensor_noise_arr2=sensor_noise_arr2,
            process_noise_value=process_noise_value,
            internal_model_noise_value=internal_model_noise_value,
            W1_cov=W1_cov,
            V1_cov=V1_cov,
            W2_cov=W2_cov,
            V2_cov=V2_cov,
            h=self.h,
            sensory_delay=self.sensory_delay,
        )

    def care_about_partner_Q(self, self_Q, partner_Q, alpha, player):
        """
        I'm implementing this in two different ways for one target vs two targets

        One Target: You pick up alpha*Partner_Q, only if that value is greater than your own Q for the ccx-ctx interaction
          - Alpha can be between [0,1] where 1 is I fully care about my partner and 0 is I don't care at all

        Two Targets: Since there are now two targets (rtx and ltx), you need to split the Q in half so the controller doesn't overshoot
          - Alpha can be between [0,0.5] where 0 is I only care about my own target and 0.5 is I care equally about both targets
          - This is effectively choosing a new target center to aim for (0.5 would be between the two targets if they split off)
          - ?? Should you split the alpha on other features of the Q matrix other than the targets?
        """
        ccx_id = self.state_mapping["ccx"]
        ccy_id = self.state_mapping["ccy"]

        ctx_id = self.state_mapping["ctx"]
        # If self Q is lower than partner Q, then take alpha*partner_Q
        self_Q[ccx_id, ccx_id] = np.maximum(np.abs(self_Q[ccx_id, ccx_id]), np.abs(partner_Q[ccx_id, ccx_id] * alpha))
        self_Q[ctx_id, ctx_id] = np.maximum(np.abs(self_Q[ctx_id, ctx_id]), np.abs(partner_Q[ctx_id, ctx_id] * alpha))
        self_Q[ccx_id, ctx_id] = -np.maximum(np.abs(self_Q[ccx_id, ctx_id]), np.abs(partner_Q[ccx_id, ctx_id] * alpha))
        self_Q[ctx_id, ccx_id] = -np.maximum(np.abs(self_Q[ctx_id, ccx_id]), np.abs(partner_Q[ctx_id, ccx_id] * alpha))
        return self_Q

    def care_about_partner_Q_two_targets(self, self_Q, partner_Q, alpha, player):
        # Can change this later, but I don't ever want to care about my partner's target more than my own
        assert alpha <= 0.5

        # TODO Make sure I'm taking on my partner's target, just like in the one target case
        # TODO I can't be adding both relevant targets, bc that makes Q huge and is overkill and doesn't match the data
        # TODO I need to put some into the partner Q, take away from other person's Q
        ccx_id = self.state_mapping["ccx"]
        ccy_id = self.state_mapping["ccy"]
        if player == 1:
            self_target_idx = self.state_mapping["rtx"]  # Care about other person's target
            self_target_idy = self.state_mapping["rty"]  # Care about other person's target
            partner_target_idx = self.state_mapping["ltx"]  # Care about other person's target
            partner_target_idy = self.state_mapping["lty"]  # Care about other person's target
        else:
            self_target_idx = self.state_mapping["ltx"]  # Care about other person's target
            self_target_idy = self.state_mapping["lty"]  # Care about other person's target
            partner_target_idx = self.state_mapping["rtx"]  # Care about other person's target
            partner_target_idy = self.state_mapping["rty"]  # Care about other person's target

        # assert self_Q[partner_target_id, partner_target_id] == 0
        # assert self_Q[ccx_id, partner_target_id] == 0
        # assert self_Q[partner_target_id, ccx_id] == 0

        # Diagonal center cursor
        #! THIS WOULD BE AN ERROR... the F_ccx + (-F_rtx) + (-F_ltx) need to add up to ZERO to hit the target
        #! SINCE THERE ARE 2 targets that would add up, we do NOT want to cut the F_ccx by (1-alpha)...
        # self_Q[ccx_id,ccx_id] = (1-alpha)*self_Q[ccx_id,ccx_id] # Need to take the proportion off of my Q
        # self_Q[ccy_id,ccy_id] = (1-alpha)*self_Q[ccy_id,ccy_id] # Need to take the proportion off of my Q

        # Diagonal term
        self_Q[partner_target_idx, partner_target_idx] = (
            alpha * self_Q[self_target_idx, self_target_idx]
        )  # How much of MY Q am I willing to give up for them
        self_Q[self_target_idx, self_target_idx] = (1 - alpha) * self_Q[
            self_target_idx, self_target_idx
        ]  # Need to take the proportion off of my Q
        # self_Q[partner_target_idy,partner_target_idy] = alpha    * self_Q[self_target_idy,self_target_idy] # How much of MY Q am I willing to give up for them
        # self_Q[self_target_idy,self_target_idy]       = (1-alpha)* self_Q[self_target_idy,self_target_idy] # Need to take the proportion off of my Q

        # Off Diagonal term
        self_Q[ccx_id, partner_target_idx] = alpha * self_Q[ccx_id, self_target_idx]
        self_Q[ccx_id, self_target_idx] = (1 - alpha) * self_Q[ccx_id, self_target_idx]

        # self_Q[ccy_id,partner_target_idy]  = alpha     * self_Q[ccy_id, self_target_idy]
        # self_Q[ccy_id,self_target_idy]     = (1-alpha) * self_Q[ccy_id, self_target_idy]

        # Off Diagonal Term
        self_Q[partner_target_idx, ccx_id] = alpha * self_Q[self_target_idx, ccx_id]
        self_Q[self_target_idx, ccx_id] = (1 - alpha) * self_Q[self_target_idx, ccx_id]

        # self_Q[partner_target_idy, ccy_id] = alpha     * self_Q[self_target_idy, ccy_id]
        # self_Q[self_target_idy, ccy_id]    = (1-alpha) * self_Q[self_target_idy, ccy_id]

        return self_Q

    def care_about_partner_Q_two_targets_v2(self, Q1, Q2, alpha1, alpha2):
        """
        Here, there's two cases
        1. Q1 = 0 and Q2 = 100
            - Then you can split Q2 between p1 and p2, proportionally
            So Q1^alpha[partner_targetx] = (1-alpha)*Q1 + alpha*Q2

        2. Q1 = 100 and Q2 = 100
            - Then, caring about your partner Q

        """
        # NOTE the Q matrix is diagonal, so we only need to access the one off-diagonal
        w1_ccx_rtx = deepcopy(Q1[..., self.state_mapping["ccx"], self.state_mapping["rtx"]])  # p1 self target weighting
        w2_ccx_ltx = deepcopy(Q2[..., self.state_mapping["ccx"], self.state_mapping["ltx"]])  # p2 self target weighting

        # * P1 caring about their partners target
        w1_ccx_ltx_alpha = alpha1 * w2_ccx_ltx  # P1 pick up a proportion of their target
        w2_ccx_ltx_alpha = (1 - alpha1) * w2_ccx_ltx  # P2 can relax a proportion of their target

        # * P2 caring about their partners target
        w2_ccx_rtx_alpha = alpha2 * w1_ccx_rtx  # P1 pick up a proportion of their target
        w1_ccx_rtx_alpha = (1 - alpha2) * w1_ccx_rtx  # P2 can relax a proportion of their target

        # * Update Q1
        # Update ccx and rtx diagonal
        Q1[..., self.state_mapping["ccx"], self.state_mapping["ccx"]] = -(w1_ccx_rtx_alpha + w1_ccx_ltx_alpha)
        Q1[..., self.state_mapping["rtx"], self.state_mapping["rtx"]] = -w1_ccx_rtx_alpha
        Q1[..., self.state_mapping["ltx"], self.state_mapping["ltx"]] = -w1_ccx_ltx_alpha
        # Update for self target
        Q1[..., self.state_mapping["ccx"], self.state_mapping["rtx"]] = w1_ccx_rtx_alpha
        Q1[..., self.state_mapping["rtx"], self.state_mapping["ccx"]] = w1_ccx_rtx_alpha
        # Help out partner
        Q1[..., self.state_mapping["ccx"], self.state_mapping["ltx"]] = w1_ccx_ltx_alpha
        Q1[..., self.state_mapping["ltx"], self.state_mapping["ccx"]] = w1_ccx_ltx_alpha

        # * Update Q2
        # update ccx and ltx
        Q2[..., self.state_mapping["ccx"], self.state_mapping["ccx"]] = -(w2_ccx_ltx_alpha + w2_ccx_rtx_alpha)
        Q2[..., self.state_mapping["ltx"], self.state_mapping["ltx"]] = -w2_ccx_ltx_alpha
        Q2[..., self.state_mapping["rtx"], self.state_mapping["rtx"]] = -w2_ccx_rtx_alpha
        # Update for self target
        Q2[..., self.state_mapping["ccx"], self.state_mapping["ltx"]] = w2_ccx_ltx_alpha
        Q2[..., self.state_mapping["ltx"], self.state_mapping["ccx"]] = w2_ccx_ltx_alpha
        # Help out partner
        Q2[..., self.state_mapping["ccx"], self.state_mapping["rtx"]] = w2_ccx_rtx_alpha
        Q2[..., self.state_mapping["rtx"], self.state_mapping["ccx"]] = w2_ccx_rtx_alpha

        # * Assertion check to make sure they will hit the arget
        if True:
            test = (
                Q1[..., self.state_mapping["ccx"], self.state_mapping["ccx"]]
                + Q2[..., self.state_mapping["ccx"], self.state_mapping["ccx"]]
                + Q1[..., self.state_mapping["rtx"], self.state_mapping["rtx"]]
                + Q1[..., self.state_mapping["ccx"], self.state_mapping["rtx"]]
                + Q1[..., self.state_mapping["rtx"], self.state_mapping["ccx"]]
                + Q2[..., self.state_mapping["rtx"], self.state_mapping["rtx"]]
                + Q2[..., self.state_mapping["ccx"], self.state_mapping["rtx"]]
                + Q2[..., self.state_mapping["rtx"], self.state_mapping["ccx"]]
                + Q1[..., self.state_mapping["ltx"], self.state_mapping["ltx"]]
                + Q1[..., self.state_mapping["ccx"], self.state_mapping["ltx"]]
                + Q1[..., self.state_mapping["ltx"], self.state_mapping["ccx"]]
                + Q2[..., self.state_mapping["ltx"], self.state_mapping["ltx"]]
                + Q2[..., self.state_mapping["ccx"], self.state_mapping["ltx"]]
                + Q2[..., self.state_mapping["ltx"], self.state_mapping["ccx"]]
            )
            assert np.all(test == 0)

        return Q1, Q2

    def add_kalman_filter(self, linear=True):
        """
        linear: True is a linear kalman filter, False is Extended (not done)
        """
        if linear:
            self.K1, self.P1_post, self.P1_prior = find_kalman_gain(
                Ad=self.MS.Ad_aug, C=self.MS.C1_aug, W=self.MS.W1, V=self.MS.V1_aug, N=self.N
            )
            self.K2, self.P2_post, self.P2_prior = find_kalman_gain(
                Ad=self.MS.Ad_aug, C=self.MS.C2_aug, W=self.MS.W2, V=self.MS.V2_aug, N=self.N
            )

    def add_feedback_gain(self, method="discrete"):
        """
        method: disc or continuous
        """
        if method == "continuous":
            out = find_coupled_feedback_gains_continuous(
                A=self.MS.A_aug,
                B1=self.MS.B1_aug,
                B2=self.MS.B2_aug,
                R11=self.MS.R11,
                R22=self.MS.R22,
                R12=self.MS.R12,
                R21=self.MS.R21,
                Q1N=self.MS.Q1_aug[-1],
                Q2N=self.MS.Q2_aug[-1],
                hit_time=self.MS.T,
                hold_steps=self.hold_steps,
                N=self.N,
                h=self.h,
                partner_knowledge=self.partner_knowledge,
            )

            (
                self.P1,
                self.P1_predict_P2,
                self.P2,
                self.P2_predict_P1,
                self.F1,
                self.F1_predict_F2,
                self.F2,
                self.F2_predict_F1,
            ) = out

        if method == "discrete":
            self.P1, self.P1_predict_P2, self.F1, self.F1_predict_F2 = find_coupled_feedback_gains_discrete(
                A=self.MS.Ad_aug,
                B1=self.MS.B1d_aug,
                B2=self.MS.B2d_aug,
                R11=self.MS.R11d,
                R22=self.MS.R22d,
                R12=self.MS.R12d,
                R21=self.MS.R21d,
                Q1=self.MS.Q1d_aug,
                Q2=self.MS.Q2d_aug,
                N=self.N,
                partner_knowledge=self.partner_knowledge,
            )
            self.P2, self.P2_predict_P1, self.F2, self.F2_predict_F1 = find_coupled_feedback_gains_discrete(
                A=self.MS.Ad_aug,
                B1=self.MS.B2d_aug,
                B2=self.MS.B1d_aug,
                R11=self.MS.R22d,
                R22=self.MS.R11d,
                R12=self.MS.R21d,
                R21=self.MS.R12d,
                Q1=self.MS.Q2d_aug,
                Q2=self.MS.Q1d_aug,
                N=self.N,
                partner_knowledge=self.partner_knowledge,
            )

    def run_simulation(self):
        (
            self.u1,
            self.u2,
            self.x,
            self.x1_post,
            self.x2_post,
            self.applied_force1,
            self.applied_force2,
            self.jump_step,
        ) = run_dual_simulation(
            N=self.N,
            h=self.h,
            x0=self.MS.x0_aug,
            Ad=self.MS.Ad_aug,
            B1d=self.MS.B1d_aug,
            B2d=self.MS.B2d_aug,
            C1=self.MS.C1_aug,
            C2=self.MS.C2_aug,
            perturbation=self.perturbation,
            probe_trial=self.probe_trial,
            probe_duration=self.probe_duration,
            state_mapping=self.state_mapping,
            K1=self.K1,
            K2=self.K2,
            F1=self.F1,
            F2=self.F2,
            process_noise_value=self.process_noise_value,
            sensor_noise_arr1=self.MS.sensor_noise_arr1,
            sensor_noise_arr2=self.MS.sensor_noise_arr2,
            internal_model_noise=self.MS.internal_model_noise_value,
            nx=self.MS.A.shape[0],
            nz=self.MS.C1.shape[0],
            perturbation_states=self.perturbation_states,
            partner_knowledge=self.partner_knowledge,
            LQR=self.LQR,
            fix_u1_val=self.fix_u1_val,
            fix_u2_val=self.fix_u2_val,
        )

        if self.jump_step < len(self.timesteps):
            self.jump_time = self.timesteps[self.jump_step]
            self.jump_back_time = self.timesteps[self.jump_step + self.probe_duration]
        else:
            self.jump_time = None
            self.jump_back_time = None

    def calculate_cost(self):
        energy_cost1 = 0
        energy_cost2 = 0
        for i in range(self.N):
            energy_cost1 += self.u1[i].T @ self.MS.R11 @ self.u1[i] + self.u2[i] @ self.MS.R12 @ self.u2[i]
            energy_cost2 += self.u2[i].T @ self.MS.R22 @ self.u2[i] + self.u1[i] @ self.MS.R21 @ self.u1[i]
        self.p1_cost = self.x[-1].T @ self.MS.Q1_aug[-1] @ self.x[-1] + energy_cost1
        self.p2_cost = self.x[-1].T @ self.MS.Q2_aug[-1] @ self.x[-1] + energy_cost2
        self.joint_cost = self.p1_cost + self.p2_cost
