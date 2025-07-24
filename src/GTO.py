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
