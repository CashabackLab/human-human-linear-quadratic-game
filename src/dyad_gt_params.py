import numpy as np
import copy
import model_functions as mf

# Timesteps
T = 0.8
h=0.01
timesteps = np.arange(0,T,h)
N = len(timesteps)
b,m = 0.1,1.5
tau = int(20*(0.001/h)) # 20 ms  
sensory_delay = 110 # ms converted to timesteps in program
probe_duration = 250 # ms, converted to timesteps in program
sensory_delay_steps = int(sensory_delay*(0.001/h))
probe_duration_steps = int(probe_duration*(0.001/h)) 
perturbation_distance = -0.03 # 3cm jumps

p1_x0, p1_y0 = 0.13, 0
p2_x0, p2_y0 = -0.13, 0
x0 = np.array([
    [p1_x0], # right px
    [0], # right vx
    [0], # right Fx
    [p1_y0], # right py
    [0], # right vy
    [0], # right Fy
    [p2_x0], # left px
    [0], # left vx
    [0], # left Fx
    [p2_y0], # left py
    [0], # left vy
    [0], # left Fy
    [((p1_x0+p2_x0)/2)], # center cursor px
    [(p1_y0+p2_y0)/2], # center cursor py
    [((p1_x0+p2_x0)/2)], # center target px
    [0.25], # center target py
])
state_mapping = {
    "rhx":0,
    "rhvx":1,
    "rfx":2,
    "rhy":3,
    "rhvy":4,
    "rfy":5,
    "lhx":6,
    "lhvx":7,
    "lfx":8,
    "lhy":9,
    "lhvy":10,
    "lfy":11,
    "ccx":12,
    "ccy":13,
    "ctx":14,
    "cty":15,
}
x = -b/m
y = 1/m
z = -1/tau
A = np.block([
    #rhx #rhvx  #rFx  #rhy  #rhvy #rFy #lhx  #lhvx #lFx  #lhy  #lhvy  #lmfy 
    [0,    1,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(4)],  # How does rhx-position change
    [0,    x,    y,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(4)],  # How does rhx-velocity change
    [0,    0,    z,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(4)],  # How does rFx change
    [0,    0,    0,   0,    1,    0,     0,   0,    0,    0,     0,    0,   np.zeros(4)],  # How does rhy-position change
    [0,    0,    0,   0,    x,    y,     0,   0,    0,    0,     0,    0,   np.zeros(4)],  # How does rhy-velocity change
    [0,    0,    0,   0,    0,    z,     0,   0,    0,    0,     0,    0,   np.zeros(4)],  # How does rFy change
    [0,    0,    0,   0,    0,    0,     0,   1,    0,    0,     0,    0,   np.zeros(4)],  # How does lhx-position change
    [0,    0,    0,   0,    0,    0,     0,   x,    y,    0,     0,    0,   np.zeros(4)],  # How does lhx-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    z,    0,     0,    0,   np.zeros(4)],  # How does lFx change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     1,    0,   np.zeros(4)],  # How does lhy-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     x,    y,   np.zeros(4)],  # How does lhy-velocity change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    z,   np.zeros(4)],  # How does lFy change
    [0,  0.5,    0,   0,    0,    0,     0, 0.5,    0,    0,     0,    0,   np.zeros(4)],  # How does cx-position change (changes based on the VELOCITY of rhxv and rhyv)
    [0,    0,    0,   0,  0.5,    0,     0,   0,    0,    0,     0.5,  0,   np.zeros(4)],  # How does cy-position change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(4)],  # How does center final target x change
    [0,    0,    0,   0,    0,    0,     0,   0,    0,    0,     0,    0,   np.zeros(4)],  # How does center final target y change
])

B1 = np.array([
    #Frx  Fry  
    [ 0,    0],  # applying to rhx-pos
    [ 0,    0],  # applying to rhx-vel
    [ 1/tau,    0],  # applying to rFx
    [ 0,    0],  # applying to rhy-pos
    [ 0,    0],  # applying to rhy-vel
    [ 0,    1/tau],  # applying to rFy
    
    [ 0,    0],  # applying to lhx-pos
    [ 0,    0],  # applying to lhx-vel
    [ 0,    0],  # applying to lFx
    [ 0,    0],  # applying to lhy-pos
    [ 0,    0],  # applying to lhy-vel
    [ 0,    0],  # applying to lFy

    [ 0,    0],  # applying to cx-pos
    [ 0,    0],  # applying to cy-pos
    
    [ 0,    0],  # applying to center target x
    [ 0,    0],  # applying to center target y
])

B2 = np.array([
    #Flx    Fly  
    [ 0,    0],  # applying to rhx-pos
    [ 0,    0],  # applying to rhx-vel
    [ 0,    0],  # applying to rFx
    [ 0,    0],  # applying to rhy-pos
    [ 0,    0],  # applying to rhy-vel
    [ 0,    0],  # applying to rFy
    
    [ 0,    0],  # applying to lhx-pos
    [ 0,    0],  # applying to lhx-vel
    [ 1/tau,    0],  # applying to lFx
    [ 0,    0],  # applying to lhy-pos
    [ 0,    0],  # applying to lhy-vel
    [ 0,    1/tau],  # applying to lFy

    [ 0,    0],  # applying to cx-pos
    [ 0,    0],  # applying to cy-pos
    
    [ 0,    0],  # applying to center target x
    [ 0,    0],  # applying to center target y
])
p1_observable_states = [
    state_mapping["rhx"], state_mapping["rhy"],
    state_mapping["rhvx"], state_mapping["rhvy"],
    state_mapping["rfx"], state_mapping["rfy"],
    state_mapping["lhx"], state_mapping["lhy"],
    state_mapping["lhvx"], state_mapping["lhvy"],
    state_mapping["ccx"], state_mapping["ccy"],
    state_mapping["ctx"], state_mapping["cty"],
]
p2_observable_states = [
    state_mapping["rhx"], state_mapping["rhy"],
    state_mapping["rhvx"], state_mapping["rhvy"],
    state_mapping["lfx"], state_mapping["lfy"],
    state_mapping["lhx"], state_mapping["lhy"],
    state_mapping["lhvx"], state_mapping["lhvy"],
    state_mapping["ccx"], state_mapping["ccy"],
    state_mapping["ctx"], state_mapping["cty"],
]
assert len(p1_observable_states) == len(p2_observable_states)

#* Create observation matrices
C1 = mf.create_observation_matrix(state_mapping, p1_observable_states)
C2 = mf.create_observation_matrix(state_mapping, p2_observable_states)

#* Base Q, for visualization puproses only of the states we care about
s=1 # should be 0 or 1 for cross term
Q = np.array(
    [
        #rhx  rhvx rfx  rhy rhvy  rFy  lhx  lhvx  lFx  lhy  lhvy lFy  ccx ccy ctx  cty
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0], # rhx 0
        [0,   1,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0], # rhvx 1
        [0,   0,   1,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0], # rFx 2
        [0,   0,   0,    s,   0,   0,    0,   0,   0,  -s,   0,   0,   0,  0,  0,  0], # rhy 3
        [0,   0,   0,    0,   1,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0], # rhvy 4
        [0,   0,   0,    0,   0,   1,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0], # rFy 5
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  0,  0,  0], # lhx 6
        [0,   0,   0,    0,   0,   0,    0,   1,   0,   0,   0,   0,   0,  0,  0,  0],  # lhvx 7
        [0,   0,   0,    0,   0,   0,    0,   0,   1,   0,   0,   0,   0,  0,  0,  0],  # lFx 8
        [0,   0,   0,   -s,   0,   0,    0,   0,   0,   s,   0,   0,   0,  0,  0,  0],  # lhy 9
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   1,   0,   0,  0,  0,  0],  # lhvy 10
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   1,   0,  0,  0,  0],  # lFy 11
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   1,  0, -1,  0], # ccx 12
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,  1,  0, -1],  # ccy 13
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,  0,    0,  -1,  0,  1,  0],  # ctx 14
        [0,   0,   0,    0,   0,   0,    0,   0,   0,   0,  0,   0,   0,  -1,  0,  1],  # cty 15
    ]
)

#* Create weights to be used in generate_Q function
w_vel = 0.2
w_force = 0.001
w_partner_vel = 0
w_rpy_lpy_cross = 0.1
Q1_WEIGHTS = {
    'rhx': 0,
    'rhvx': w_vel,
    'rfx': w_force,
    'rhy': w_rpy_lpy_cross,
    'rhvy': w_vel,
    'rfy': w_force,
    'lhx': 0,
    'lhvx': 0,
    'lfx': 0,
    'lhy': w_rpy_lpy_cross,
    'lhvy': 0,
    'lfy': 0,
    'ccx': 1,
    'ccy': 1,
    'ctx': 1,
    'cty': 1,
}
Q2_WEIGHTS = {
    'rhx': 0,
    'rhvx': 0,
    'rfx': 0,
    'rhy': w_rpy_lpy_cross,
    'rhvy': 0,
    'rfy': 0,
    'lhx': 0,
    'lhvx': w_vel,
    'lfx': w_force,
    'lhy': w_rpy_lpy_cross,
    'lhvy': w_vel,
    'lfy': w_force,
    'ccx': 1,
    'ccy': 1,
    'ctx': 1,
    'cty': 1,
}

QVAL = 4e4
IRREL_QVAL = 100
RVAL = 1e-5
help_perc = 0.5

R11 = np.eye(B1.shape[1])*RVAL # Square Enery cost matrix with dimensions according to the columns of B
R22 = np.eye(B2.shape[1])*RVAL # Square Enery cost matrix with dimensions according to the columns of B
R12 = np.eye(B1.shape[1])*RVAL # Square Enery cost matrix with dimensions according to the columns of B
R21 = np.eye(B2.shape[1])*RVAL # Square Enery cost matrix with dimensions according to the columns of B

#* Create noise and covariance matrices
# From Nashed 2012 VARIANCES: [process_noise = 10e-2, feedback/sensory_noise=10e-6, internal_model_noise=10e-6]
# sig_pos = np.sqrt(10e-6)
# sig_vel = np.sqrt(10e-6)
# sig_f = np.sqrt(10e-6)
INTERNAL_MODEL_NOISE = 1e-5 # Should only affect current prediction, not previous augmented ones
PROCESS_NOISE = 1e-2
# # From Todorov OFC adapted to noise characterstics
# sig_s = 0.25 # 0.5 from todorov 
# sig_pos = (0.02*sig_s)**2
# sig_vel = (0.2*sig_s)**2
# sig_f = (0.1*sig_s)**2
sig_pos = 0.0001
sig_vel = 0.001
sig_f = 0.1
SENSOR_NOISE_ARR1 =  np.array([
    sig_pos, # right px
    sig_vel, # right vx
    sig_f, # right fx
    sig_pos, # right py
    sig_vel, # right vy
    sig_f, # right fy
    sig_pos, # left px
    sig_vel, # left vx
    sig_pos, # left py
    sig_vel, # left vy
    sig_pos, # center cursor px
    sig_pos, # center cursor py
    sig_pos, # center target px
    sig_pos, # center target py
])
SENSOR_NOISE_ARR2 =  np.array([
    sig_pos, # right px
    sig_vel, # right vx
    sig_pos, # right py
    sig_vel, # right vy
    sig_pos, # left px
    sig_vel, # left vx
    sig_f, # left fx
    sig_pos, # left py
    sig_vel, # left vy
    sig_f, # left fy
    sig_pos, # center cursor px
    sig_pos, # center cursor py
    sig_pos, # center target px
    sig_pos, # center target py
])

sensor_cov = 0.01
W1_cov =  np.diag(np.array([
    sensor_cov, # right px
    sensor_cov, # right vx
    sensor_cov, # right fx
    sensor_cov, # right py
    sensor_cov, # right vy
    sensor_cov, # right fy
    sensor_cov, # left px
    sensor_cov, # left vx
    sensor_cov, # left py
    sensor_cov, # left vy
    sensor_cov, # center cursor px
    sensor_cov, # center cursor py
    sensor_cov, # center target px
    sensor_cov, # center target py
]))
W2_cov =  np.diag(np.array([
    sensor_cov, # right px
    sensor_cov, # right vx
    sensor_cov, # right py
    sensor_cov, # right vy
    sensor_cov, # left px
    sensor_cov, # left vx
    sensor_cov, # left fx
    sensor_cov, # left py
    sensor_cov, # left vy
    sensor_cov, # left fy
    sensor_cov, # center cursor px
    sensor_cov, # center cursor py
    sensor_cov, # center target px
    sensor_cov, # center target py
]))

process_cov = 0.0003
process_adj = 100 # Higher process noise on partner, trust sensory info more for them
V1_cov =  np.diag(np.array([
    process_cov, # right px
    process_cov, # right vx
    process_cov, # right fx
    process_cov, # right py
    process_cov, # right vy
    process_cov, # right fy
    process_cov*process_adj, # left px
    process_cov*process_adj, # left vx
    process_cov*process_adj, # left fx
    process_cov*process_adj, # left py
    process_cov*process_adj, # left vy
    process_cov*process_adj, # left fy
    process_cov, # center cursor px
    process_cov, # center cursor py
    process_cov, # center target px
    process_cov, # center target py
]))

V2_cov =  np.diag(np.array([
    process_cov*process_adj, # right px
    process_cov*process_adj, # right vx
    process_cov*process_adj, # right fx
    process_cov*process_adj, # right py
    process_cov*process_adj, # right vy
    process_cov*process_adj, # right fy
    process_cov, # left px
    process_cov, # left vx
    process_cov, # left fx
    process_cov, # left py
    process_cov, # left vy
    process_cov, # left fy
    process_cov, # center cursor px
    process_cov, # center cursor py
    process_cov, # center target px
    process_cov, # center target py
]))

# Same order as const.condition_names and to match p1 for const.collapsed_condition_names
player_relevancy = [
    ("irrelevant","irrelevant"),
    ("relevant","irrelevant"),
    ("irrelevant","relevant"),
    ("relevant","relevant"),
]

help_percentages = [0.0, 0.0, 1.0, 0.5]
partner_knowledge = [False, True, True, True]