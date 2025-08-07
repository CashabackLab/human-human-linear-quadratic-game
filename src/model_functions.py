import numpy as np
from scipy.linalg import pinv
from scipy.integrate import solve_ivp
from copy import deepcopy
import polars as pl 

def augment_A_matrix(A,sensory_delay):
    out = np.block([
            [A , np.zeros((A.shape[0], A.shape[1]*sensory_delay))],
            [np.eye(A.shape[0]*sensory_delay,A.shape[1]*sensory_delay), np.zeros((A.shape[0]*sensory_delay,A.shape[1]))],
    ])
    return out

def augment_Q_matrix(Q, sensory_delay):
    # Using negatives to index bc so we can have the first index be time for a 3D time-varying Q
    if Q.ndim == 2:
        out = np.block([
                    [Q,   np.zeros((Q.shape[0], sensory_delay*Q.shape[1]))],
                    [np.zeros((Q.shape[0]*sensory_delay,Q.shape[1])), np.zeros((Q.shape[0]*sensory_delay, Q.shape[1]*sensory_delay)) ]
                ])
    else:
        out = np.block([
                    [Q,   np.zeros((Q.shape[0], Q.shape[1], sensory_delay*Q.shape[2]))],
                    [np.zeros((Q.shape[0], Q.shape[1]*sensory_delay,Q.shape[2])), np.zeros((Q.shape[0], Q.shape[1]*sensory_delay, Q.shape[2]*sensory_delay)) ]
                ])
        
    return out

def augment_B_matrix(B, sensory_delay):
    out = np.block([
                [B],
                [np.zeros((B.shape[0]*sensory_delay, B.shape[1]))]
            ])
    return out

def generate_Q(weight_dict, state_mapping, cross_terms:list[tuple], QVAL):
    # If k is in cross_terms, then leave diagonal as 1 and do the cross_terms together
    assert len(weight_dict) == len(state_mapping)
    assert list(weight_dict.keys()) == list(state_mapping.keys())
    
    Q = np.zeros((len(state_mapping),len(state_mapping)))
    for k,v in weight_dict.items():
        # Check for Off-Diagonals
        pair_check = [pair for pair in cross_terms if k in pair] 
        if len(pair_check)>0:
            for pair in pair_check:
                k1,k2 = pair
                Q[state_mapping[k1], state_mapping[k2]] = -v*QVAL
                Q[state_mapping[k2], state_mapping[k1]] = -v*QVAL
        # else:
        Q[state_mapping[k], state_mapping[k]] = v*QVAL
    return Q

def create_observation_matrix(state_mapping, observable_states):
    nx = len(state_mapping) # Num states
    nz = len(observable_states) # num sensory states
    C = np.zeros((nz,nx))
    row = 0
    for key,idx in state_mapping.items():
        if idx in observable_states:
            C[row, idx] = 1
            row+=1
    return C

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

class Models:
    def __init__(self, regular_df:pl.DataFrame, probe_df:pl.DataFrame,
                 alphas:list[float] = [0.0, 0.0, 1.0, 0.5], 
                 partner_knowledges:list[bool] = [False, True, True, True]):
        self.regular_df = regular_df
        self.probe_df = probe_df.with_columns(pl.col(pl.Float64()).round(3))
        # self.models = models
        
        self.alphas = alphas
        self.partner_knowledges = partner_knowledges
        self.max_exp_trial_number = probe_df["experiment_trial_number"].max()
        self.cropped_probe_df = self._crop_probe_df()
        
    def _crop_probe_df(self):
        before_probe_time = 0.05 # 50ms
        after_probe_time = 0.4 # 400ms
        extra_time = 0 # 100ms so that none of the models get cut off and can't vstack below
        time_from_probe_onset = np.arange(-before_probe_time, after_probe_time+0.01, 0.01).round(2)*1000
        #* Crop probe df
        df_list = []
        for alpha, partner_knowledge in zip(self.alphas, self.partner_knowledges):
            for i in range(1,self.max_exp_trial_number+1):
                dff = self.probe_df.filter(
                    pl.col("alpha") == alpha,
                    pl.col("partner_knowledge") == partner_knowledge,
                    pl.col("experiment_trial_number") == i
                )
                start_index = np.round(dff["jump_time"][0] - before_probe_time,2) # Need to round so that the is_between works correctly
                end_index = np.round(dff["jump_time"][0] + after_probe_time + extra_time,2)
                filt_df = dff.filter(pl.col("timepoint").is_between(start_index, end_index, closed="both"))
                # assert len(filt_df) == 46 # 50ms before and 400ms after the probe onset
                #* Add time from probe onset column
                filt_df = filt_df.with_columns(
                    time_from_probe_onset=time_from_probe_onset,
                    p1_applied_force_double = pl.col("p1_applied_force").mul(2)
                )
                # if len(filt_df)<47:
                #     print(alpha, partner_knowledge, i)
                df_list.append(filt_df)
        return pl.concat(df_list) 
    
    def lateral_deviation(self):
        pass
    
    def involuntary_feedback_response(self):
        pass
    
    def onset_times(self):
        pass