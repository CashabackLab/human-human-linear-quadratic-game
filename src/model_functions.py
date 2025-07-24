import numpy as np

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