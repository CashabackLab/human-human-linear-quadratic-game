import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import polars as pl
import dill
from pathlib import Path
import data_visualization as dv
from time import time
import copy
import importlib
import LQG
import dyad_lqg_muscle_params_v2
import constants as const
import plot_functions as pf
import helper_functions as hf

importlib.reload(LQG)
importlib.reload(dyad_lqg_muscle_params_v2)
import LQG
import dyad_lqg_muscle_params_v2 as param
wheel = dv.ColorWheel()
plt.style.use("cashaback_dark")
'''

Comparing models using the differential game theory model.

The GT model is just an LQG if partner_knowledge = False.
It's also a bimanual model if parnter_knowledge = True and alpha = 1.0.
It's completely selfish if partner_knowledge = True and alpha = 0.0.

'''
#%% Functions
def create_single_trial_df(trial, model, state_mapping, condition, p1_target, p2_target, experiment_trial_number):
    df = pl.DataFrame()
    df = df.with_columns(
        timepoint = model.state_timesteps,
        condition = np.array([condition]*len(model.state_timesteps)),
        p1_target = np.array([p1_target]*len(model.state_timesteps)),
        p2_target = np.array([p2_target]*len(model.state_timesteps)),
        trial = np.array([trial]*len(model.state_timesteps)),
        experiment_trial_number = np.array([experiment_trial_number]*len(model.state_timesteps)),
        jump_type = np.array([model.perturbation_states[0]]*len(model.state_timesteps)),
        jump_time = np.array([model.jump_time]*len(model.state_timesteps)),
        jump_back_time = np.array([model.jump_back_time]*len(model.state_timesteps)),
        alpha = np.array([model.alpha]*len(model.state_timesteps),dtype=float),
        partner_knowledge = np.array([model.partner_knowledge]*len(model.state_timesteps)),
        p1_applied_force = np.append(np.array(model.applied_force1), np.nan),
        p2_applied_force = np.append(np.array(model.applied_force2), np.nan),
        u1x = np.append(model.u1[:,0],np.nan),
        u1y = np.append(model.u1[:,1],np.nan),
        u2x = np.append(model.u2[:,0],np.nan),
        u2y = np.append(model.u2[:,1],np.nan),
    )
    for key, val in state_mapping.items():
        df = df.with_columns(pl.Series(key, model.x[:,val]))
    return df
#%%
SAVE = True
SAVE_PATH = Path(r"..\..\data\models")
#%% Set up constants and matrices
# Timesteps
HIT_TIME = 0.8
HOLD_TIME = 0
h=0.01
sensory_delay = int(param.sensory_delay*(0.001/h))
perturbation = param.perturbation_distance # Perturabtion distance
probe_duration = int(param.probe_duration*(0.001/h)) 
QVAL = param.QVAL
IRREL_QVAL = param.IRREL_QVAL
RVAL = param.RVAL
B_bimanual = np.hstack((param.B1,param.B2))

state_mapping = copy.deepcopy(param.state_mapping)
#%% Q and R initialize
Q1 = hf.generate_Q(
    weight_dict=param.Q1_WEIGHTS, 
    state_mapping=param.state_mapping, 
    cross_terms=[["ccx","ctx"], ["ccy","cty"], 
                 ["rhy","lhy"]], 
    QVAL=QVAL
)
Q2 = hf.generate_Q(
    weight_dict=param.Q2_WEIGHTS, 
    state_mapping=param.state_mapping, 
    cross_terms=[["ccx","ctx"], ["ccy","cty"],
                    ["rhy","lhy"]], 
    QVAL = QVAL,
)

Q1_xvals = [
    (IRREL_QVAL,-IRREL_QVAL,-IRREL_QVAL), # Joint Irrelevant
    (QVAL,-QVAL,-QVAL), # P1 Relevant
    (IRREL_QVAL,-IRREL_QVAL,-IRREL_QVAL), # P2 Relevant
    (QVAL,-QVAL,-QVAL), # Joint Relevant
]
Q2_xvals = [
    (IRREL_QVAL,-IRREL_QVAL,-IRREL_QVAL), # Joint Irrelevant
    (IRREL_QVAL,-IRREL_QVAL,-IRREL_QVAL), # P1 Relevant
    (QVAL,-QVAL,-QVAL), # P2 Relevant
    (QVAL,-QVAL,-QVAL), # Joint Relevant
]

R_bimanual = np.eye(B_bimanual.shape[1])*RVAL
R11 = np.eye(param.B1.shape[1])*RVAL # Square Enery cost matrix with dimensions according to the columns of B
R22 = np.eye(param.B2.shape[1])*RVAL # Square Enery cost matrix with dimensions according to the columns of B

help_percentages = [0.0, 0.0, 1.0, 0.5]
partner_knowledge = [False, True, True, True]
jump_type = "ccx"
if jump_type == "ctx":
    perturbation = -perturbation
player_relevancy = param.player_relevancy
regular_models = {}
probe_models = {}
num_trials = 100
regular_df_list = []
probe_df_list = []
for i in range(len(help_percentages)):
    print(f"model {i+1}")
    c = 0 # experiment trial number counter
    for j in range(len(const.condition_names)):
        Q1[state_mapping['ccx'],state_mapping['ccx']] = Q1_xvals[j][0]
        Q1[state_mapping['ccx'],state_mapping['ctx']] = Q1_xvals[j][1]
        Q1[state_mapping['ctx'],state_mapping['ccx']] = Q1_xvals[j][2]
        Q2[state_mapping['ccx'],state_mapping['ccx']] = Q2_xvals[j][0]
        Q2[state_mapping['ccx'],state_mapping['ctx']] = Q2_xvals[j][1]
        Q2[state_mapping['ctx'],state_mapping['ccx']] = Q2_xvals[j][2]
        
        model = LQG.GameTheoryLQG(
            name = const.condition_names[j],
            p1_target=player_relevancy[j][0],
            p2_target=player_relevancy[j][1],
            A = param.A, 
            B1 = param.B1,
            B2 = param.B2,
            Q1 = Q1,
            Q2 = Q2,
            R11 = R11,
            R22 = R22,
            R12 = R22*help_percentages[i],
            R21 = R11*help_percentages[i],
            C1 = param.C1, 
            C2 = param.C2, 
            x0 = param.x0, # Initial state, 
            sensor_noise_arr1 = param.SENSOR_NOISE_ARR1,
            sensor_noise_arr2 = param.SENSOR_NOISE_ARR2,
            process_noise_value =  param.PROCESS_NOISE,
            internal_model_noise_value=param.INTERNAL_MODEL_NOISE,
            W1_cov = param.W1_cov,
            W2_cov = param.W2_cov,
            V1_cov = param.V1_cov,
            V2_cov = param.V2_cov,
            perturbation=perturbation,
            sensory_delay=sensory_delay,
            h=h,
            HIT_TIME=HIT_TIME,
            HOLD_TIME = HOLD_TIME,
            state_mapping = state_mapping,
            partner_knowledge=partner_knowledge[i],
            probe_trial=False,
            LQR=False,
            probe_duration=probe_duration,
            alpha=help_percentages[i],
            perturbation_states=jump_type,
        )
        
        # Q vals should be the same for both players in Joint Irrelevant and Joint Relevant conditions
        # for ccx position and on (ccx, ccy, ctx, cty)
        idx = state_mapping['ccx']
        if j==0 or j==3:
            assert np.all(model.MS.Q1[idx:,idx:] == model.MS.Q2[idx:,idx:])
            
        model.add_kalman_filter()
        model.add_feedback_gain()
        
        probe_model = copy.deepcopy(model)
        probe_model.probe_trial = True
        if SAVE:
            with open(SAVE_PATH / f"regular_p1_{player_relevancy[j][0]}_p2_{player_relevancy[j][1]}_"
                                    f"jumptype_{jump_type}_" 
                                    f"perturbation_{perturbation}_"
                                    f"alpha_{help_percentages[i]}_"
                                    f"partnerknowledge_{partner_knowledge[i]}.pkl","wb") as f:
                dill.dump(model,f)
            with open(SAVE_PATH / f"probe_p1_{player_relevancy[j][0]}_p2_{player_relevancy[j][1]}_"
                                    f"jumptype_{jump_type}_"
                                    f"perturbation_{perturbation}_"
                                    f"alpha_{help_percentages[i]}_"
                                    f"partnerknowledge_{partner_knowledge[i]}.pkl","wb") as f:
                dill.dump(probe_model,f)
        #* Create long format data for trials
        fig = dv.AutoFigure('a')
        for k in range(num_trials):
            c+=1
            model.run_simulation()
            model.calculate_cost()
            probe_model.run_simulation()
            probe_model.calculate_cost()
            # print(probe_model.applied_force1)
            regular_df = create_single_trial_df(trial=k+1, model=model,
                                    state_mapping=state_mapping, condition=model.name,
                                    p1_target = model.p1_target,
                                    p2_target = model.p2_target, 
                                    experiment_trial_number = c)
            regular_df_list.append(regular_df)
            
            probe_df = create_single_trial_df(trial=k+1, model=probe_model,
                                                state_mapping=state_mapping, condition=probe_model.name,
                                                p1_target = probe_model.p1_target,
                                                p2_target = probe_model.p2_target, 
                                                experiment_trial_number=c)
            probe_df_list.append(probe_df)

            #* Plotting
            rhx_id,rhy_id = state_mapping["rhx"], state_mapping["rhy"]
            lhx_id,lhy_id = state_mapping["lhx"],state_mapping["lhy"]
            ccx_id,ccy_id = state_mapping["ccx"],state_mapping["ccy"]
            ctx_id,cty_id = state_mapping["ctx"],state_mapping["cty"]
            
            ax = fig.axes['a']
            ax.scatter(model.x[:, rhx_id],model.x[:, rhy_id],color=wheel.grey,marker="x",s=10,)  # Actual hand position scattered
            ax.plot(model.x2_post[:, rhx_id], model.x2_post[:, rhy_id], color=wheel.pink)  # Posterior estiamte of the cursor

            ax.scatter(model.x[:, lhx_id],model.x[:, lhy_id],color=wheel.grey,marker="x",s=10,)  # Actual hand position scattered
            ax.plot(model.x2_post[:, lhx_id], model.x2_post[:, lhy_id], color=wheel.rak_blue)  # Posterior estiamte of the cursor

            ax.scatter(model.x[:, ccx_id],model.x[:, ccy_id],color=wheel.grey,marker="x",s=10,)  # Posterior estiamte of the center cursor
            ax.plot(model.x2_post[:, ccx_id],model.x2_post[:, ccy_id],color=wheel.sunflower,)  # Posterior estiamte of the center cursor
            ax.scatter(model.x[-1,ctx_id], model.x[-1,cty_id], c=wheel.sunflower)
        plt.show()
        
        regular_models[const.condition_names[j]] = copy.deepcopy(model)
        probe_models[const.condition_names[j]] = copy.deepcopy(probe_model)

regular_df = pl.concat(regular_df_list)
probe_df = pl.concat(probe_df_list)
#%%
if SAVE:
    with open(SAVE_PATH / f"regular_model_{jump_type}_jump_df.pkl","wb") as f:
        dill.dump(regular_df.with_row_index(),f)
        
    with open(SAVE_PATH / f"probe_model_{jump_type}_jump_df.pkl","wb") as f:
        dill.dump(probe_df.with_row_index(),f)


#%% Plot Feedback Gains
fig = dv.AutoFigure("ab;cd", figsize=(9,9), dpi=120)
axes = list(fig.axes.values())
for j,condition in enumerate(const.condition_names):
    model = regular_models[condition]
    pf.plot_feedback_gains(ax=axes[j],F1=model.F1,F2=model.F2, 
                            state_mapping=state_mapping, 
                            state_labels=["ccx","ccy"],
                            timesteps=model.timesteps,
                            )
    axes[j].set_title(condition)
    # axes[j].set_ylim(-10,18000)
    axes[j].axvline(model.jump_time)
    axes[j].axvline(model.jump_back_time)
    axes[j].axvline(model.jump_time + 0.12, c=wheel.grey)
    axes[j].axvline(model.jump_back_time + 0.12, c=wheel.grey)
    
#%% Plot xy states
fig = dv.AutoFigure("ab;cd", figsize=(6,6), dpi=120)
axes = list(fig.axes.values())
for j,condition in enumerate(const.condition_names):
    model = regular_models[condition]
    axes[j].plot(model.x[:,state_mapping["rhx"]], model.x[:,state_mapping["rhy"]],c=const.cursor_colors["p1"])
    axes[j].scatter(model.x[:,state_mapping["ccx"]], model.x[:,state_mapping["ccy"]],c="grey",marker="x")
    axes[j].plot(model.x1_post[:,state_mapping["ccx"]], model.x1_post[:,state_mapping["ccy"]],c=const.cursor_colors["cc"])
    axes[j].plot(model.x[:,state_mapping["lhx"]], model.x[:,state_mapping["lhy"]],c=const.cursor_colors["p2"])
    axes[j].scatter(model.x[0,state_mapping["ctx"]], model.x[0,state_mapping["cty"]],facecolor="None", edgecolors="grey")
    axes[j].set_title(condition, fontsize=9)
    # axes[j].set_xlim(-.2,.2)
    # axes[j].set_ylim(-.01,.3)

# fig.fig.suptitle(model_name)

#%% Plot Y-pos over time
fig = dv.AutoFigure("ab;cd", figsize=(9,9), dpi=120)
axes = list(fig.axes.values())
for j,condition in enumerate(const.condition_names):
    model = regular_models[condition]
    pf.plot_states_over_time(ax=axes[j],x=model.x, 
                                state_mapping=state_mapping, 
                                state_labels=["rhy","lhy","ccy"],
                                linestyles=["-","--",":"])
    axes[j].set_title(condition)
    axes[j].set_ylabel("Y-Position (m)")
#%% Plot X-pos over time
fig = dv.AutoFigure("ab;cd", figsize=(9,9), dpi=120)
axes = list(fig.axes.values())
for j,condition in enumerate(const.condition_names):
    model = probe_models[condition]
    pf.plot_states_over_time(ax=axes[j],x=model.x1_post, 
                                state_mapping=state_mapping, 
                                state_labels=["rhx","lhx","ccx"],
                                linestyles=["-","-",":"])
    axes[j].set_title(condition)
    axes[j].set_ylabel("X-Position (m)")
#%% Plot X-Velocity over time
fig = dv.AutoFigure("ab;cd", figsize=(9,9), dpi=120)
axes = list(fig.axes.values())
for j,condition in enumerate(const.condition_names):
    model = regular_models[condition]
    pf.plot_states_over_time(ax=axes[j],x=model.x, 
                                state_mapping=state_mapping, 
                                state_labels=["rhvx","lhvx"],
                                linestyles=["-","-",":"])
    axes[j].set_title(condition)
    axes[j].set_ylabel("X-Velocity (m/s)")
    axes[j].set_ylim(-0.01,0.2)
#%% Plot force responses
fig = dv.AutoFigure("ab;cd", figsize=(9,9), dpi=120)
axes = list(fig.axes.values())
ylim = (-0.1,2.5)
for j,condition in enumerate(const.condition_names):
    model = probe_models[condition]
    axes[j].plot(model.timesteps, model.applied_force1, color=wheel.rak_red, label="RH Lateral Force") 
    axes[j].plot(model.timesteps, model.applied_force2, color=wheel.rak_blue, label="LH Lateral Force", ls='--') 
    axes[j].axvline(x=model.jump_time, ls='--', color=wheel.grey)
    axes[j].axvline(x=model.jump_back_time, ls='--', color=wheel.grey)

    # PLot involuntary region
    axes[j].fill_betweenx(np.arange(ylim[0],ylim[1],0.01), 
                          model.jump_time + 0.18, # add 180ms, relative to total time
                          model.jump_time + 0.23, # add 230ms, relative to total time
                          facecolor = wheel.lighten_color(wheel.light_grey,0.75), alpha = 0.1)
    axes[j].text(model.jump_time + 0.205, ylim[1],"Involuntary", rotation=90, va='top', ha='center', color=wheel.grey, fontsize=10)
    

    axes[j].set_xticks(np.arange(0,HIT_TIME+0.1,0.1))
    axes[j].set_ylabel("Applied Force Into Channel")
    axes[j].set_xlabel("Time (ms)")
    axes[j].set_title(condition)
    # axes[j].set_xlim(model.jump_time-0.02,0.7)
    axes[j].set_ylim(ylim[0],ylim[1])
    
    axes[j].legend()
#%% One plot all condition force traces
fig = dv.AutoFigure("a", figsize=(6.5,5), dpi=300)
ax = fig.axes['a']
ylim = (-0.1,6)
for j,condition in enumerate(const.condition_names):
    model = probe_models[condition]
    ax.plot(model.timesteps, 2*model.applied_force2, color=const.condition_colors_dark[j], label=const.collapsed_condition_names[j]) 
    # ax.plot(model.timesteps, model.applied_force2, color=const.condition_colors_light[j], label=const.collapsed_condition_names[j]) 
    ax.axvline(x=model.jump_time, ls='--', color=wheel.grey)
    ax.axvline(x=model.jump_back_time, ls='--', color=wheel.grey)

    # peakx = model.timesteps[60]
    # peaky = np.max(2*model.applied_force1)
    # ax.text(peakx, peaky,const.collapsed_condition_names[j], 
    #         transform=ax.transData, color=const.condition_colors_dark[j], 
    #         va='bottom', ha='center', fontsize=6.5, fontweight='bold')
    
    # PLot involuntary region
    ax.fill_betweenx(np.arange(ylim[0],ylim[1],0.01), 
                          model.jump_time + 0.18, # add 180ms, relative to total time
                          model.jump_time + 0.23, # add 230ms, relative to total time
                          facecolor = wheel.lighten_color(wheel.light_grey,0.75), alpha = 0.1)
    ax.text(model.jump_time + 0.205, ylim[1],"Involuntary", rotation=90, va='top', ha='center', color=wheel.grey, fontsize=10)
    

    # ax.set_xticks(np.arange(0,T+0.1,0.1))
    ax.set_xticks(np.linspace(model.jump_time - 0.050, # 50ms before jump
                        model.jump_time+0.400,
                        10),labels = np.linspace(-50,400,10))
    ax.set_ylabel("Applied Force Into Channel")
    ax.set_xlabel("Time (ms)")
    # ax.set_title(condition)
    ax.set_xlim(model.jump_time-0.02,model.jump_time+0.4)
    ax.set_ylim(ylim[0],ylim[1])

    handles, labels = fig.axes['a'].get_legend_handles_labels()
    ax.legend(handles, labels,ncols=1,loc=(0.2,0.5), 
              labelcolor="linecolor",frameon=False)
    
    
#%% Involuntary Force
inv_start = np.argwhere(model.timesteps == (model.jump_time+0.18).round(3))[0,0]
inv_end = np.argwhere(model.timesteps == (model.jump_time+0.23).round(3))[0,0]
vol_start = np.argwhere(model.timesteps == (model.jump_time+0.3).round(3))[0,0]
vol_end = np.argwhere(model.timesteps == (model.jump_time+0.4).round(3))[0,0]

both_data = [[np.mean(model.applied_force2[inv_start:inv_end]) for model in probe_models.values()],
             [np.mean(model.applied_force2[vol_start:vol_end]) for model in probe_models.values()]]
ylabels = ["Involuntary Visuomotor\nFeedback Response (N)", 
           "Voluntary Visuomotor\nFeedback Response (N)"]
titles = ["Involuntary Feedback Response", 
          "Voluntary Feedback Response"]
ylims = [(-0.2,4), (-0.2,6)]
yvals = [[1.5,2.5,2.9,3.3],
         [4.6,7.1,7.4,8]
]
for i in range(2):
    mosaic="a"
    fig = dv.AutoFigure(
        mosaic=mosaic, 
        dpi=150,
    )
    
    ax    = fig.axes['a']
    #* Boxplots
    xlocs = np.arange(0,4,1)
    bw    = 0.3
    for j,condition in enumerate(const.collapsed_condition_names):
        dv.boxplot(ax, xlocs[j], 2*both_data[i][j], jitter_data=False, 
                color=const.condition_colors_dark[j], 
                box_width=bw)
    #*
    ax.set_xticks(xlocs)
    ax.set_xticklabels(const.collapsed_condition_names, fontweight="bold")
    [tick.set_color(const.condition_colors_dark[k]) for k,tick in enumerate(ax.xaxis.get_ticklabels())]
    ax.tick_params(axis='both', which='major', labelsize=9)

    ax.set_ylabel(ylabels[i], fontsize=10)
    # ax.set_title(titles[i], fontsize=12)
    ax.set_ylim(ylims[i])
    ax.spines[["bottom"]].set_visible(False)
    ax.tick_params(bottom=False)
    fig.remove_figure_borders()

#%% Plot Costs
mosaic="a"
fig = dv.AutoFigure(
    mosaic=mosaic, 
    dpi=150,
)
ax = fig.axes['a']
#* Boxplots
xlocs = np.arange(0,4,1)
bw    = 0.3
for j,condition in enumerate(const.condition_names):
    model = regular_models[condition]
    ax.bar(xlocs[j], model.p1_cost, bottom=0, 
            color=const.cursor_colors["p1"], 
            width=bw, label="p1")
    ax.bar(xlocs[j], model.p2_cost, bottom=model.p1_cost, 
            color=const.cursor_colors["p2"], 
            width=bw, label="p2")
#*
ax.set_xticks(xlocs)
ax.set_xticklabels(const.condition_names, fontweight="bold")
[tick.set_color(const.condition_colors_dark[k]) for k,tick in enumerate(ax.xaxis.get_ticklabels())]
ax.tick_params(axis='both', which='major', labelsize=9)

ax.set_ylabel("Cost (au)", fontsize=10)
# ax.set_title(titles[i], fontsize=12)
# ax.set_ylim(ylims[i])
ax.spines[["bottom"]].set_visible(False)
ax.tick_params(bottom=False)
dv.custom_legend(ax, labels=["p1","p2"], colors=[const.cursor_colors["p1"],
                                             const.cursor_colors["p2"]],
                 loc="upper left")
fig.remove_figure_borders()

#%% Plot Kalman Gains
# state_labels = ["rhx","rhy","ccx","ccy"]
# fig = dv.AutoFigure("ab;cd", figsize=(9,9), dpi=120)
# axes = list(fig.axes.values())
# for j,condition in enumerate(const.condition_names):
#     model = regular_models[condition]
#     for label in state_labels:
#         axes[j].plot(model.state_timesteps,model.K1[:,state_mapping[label],state_mapping[label]], label=f"P1 K {label}")
#         axes[j].plot(model.state_timesteps,model.K2[:,state_mapping[label],state_mapping[label]], label=f"P2 K {label}")
#         axes[j].set_ylabel("K(t)")
#         axes[j].set_xlabel("Time (s)")
#         axes[j].legend(loc="best")
#     axes[j].set_title(condition)
#     axes[j].set_ylim(-0.1,1)

