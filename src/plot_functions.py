import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import constants as const
import data_visualization as dv
import constants as const
import matplotlib.animation as animation

wheel = dv.ColorWheel()


def get_target_patches(ax, self_color=const.self_color, partner_color=const.partner_color, flip_p2=True, 
                       border_width = 1, target_jump=False, alpha=None, edge_alpha=1, face_alpha=1, x1=None, y1=None, x2=None, y2=None):
    if alpha is not None:
        edge_alpha = alpha
        face_alpha = alpha
        
    if flip_p2:
        translation=0
    else:
        translation = 0.54*2
    
    if target_jump:
        pert = 0.03
    else:
        pert = 0
        
    if x1 is None:
        x_rel = const.rel_target_x_corner
        x_irrel = const.irrel_target_x_corner
    else:
        x_rel = x1
        x_irrel = x1
        
    if y1 is None:
        y_rel = const.rel_target_y_corner
        y_irrel = const.rel_target_y_corner
    else:
        y_rel = y1
        y_irrel = y1
    

    p1_targets = {
        "joint_irrelevant": mpl.patches.Rectangle(
            (x_irrel+pert, y_irrel),
            const.irrel_target_width,
            const.target_height,
            edgecolor=(*wheel.hex_to_rgb(self_color, normalize=True),edge_alpha),
            facecolor=(*wheel.hex_to_rgb(self_color, normalize=True),face_alpha),
            lw=0,
            clip_on=False,
            transform=ax.transData,
        ),
        "p1_relevant": mpl.patches.Rectangle(
            (x_rel+pert, y_rel),
            const.rel_target_width,
            const.target_height,
            edgecolor=(*wheel.hex_to_rgb(self_color, normalize=True),edge_alpha),
            facecolor=(*wheel.hex_to_rgb(self_color, normalize=True),face_alpha),
            lw=border_width,
            clip_on=False,
            transform=ax.transData,
            # alpha=alpha,
        ),
        "p2_relevant": mpl.patches.Rectangle(
            (x_irrel+pert, y_irrel),
            const.irrel_target_width,
            const.target_height,
            edgecolor=(*wheel.hex_to_rgb(self_color, normalize=True),edge_alpha),
            facecolor=(*wheel.hex_to_rgb(self_color, normalize=True),face_alpha),
            lw=border_width,
            clip_on=False,
            transform=ax.transData,
        ),
        "joint_relevant": mpl.patches.Rectangle(
            (x_rel+pert, y_rel),
            const.rel_target_width,
            const.target_height,
            edgecolor=(*wheel.hex_to_rgb(self_color, normalize=True),edge_alpha),
            facecolor=(*wheel.hex_to_rgb(self_color, normalize=True),face_alpha),
            lw=0,
            clip_on=False,
            transform=ax.transData,
        ),
    }
    p2_targets_copy = {
        "joint_irrelevant": mpl.patches.Rectangle(
            (x_irrel+pert-translation, y_irrel),
            const.irrel_target_width,
            const.target_height,
            edgecolor=(*wheel.hex_to_rgb(partner_color, normalize=True),edge_alpha),
            facecolor="none",
            lw=border_width,
            clip_on=False,
            transform=ax.transData,
        ),
        "p1_relevant": mpl.patches.Rectangle(
            (x_irrel+pert-translation, y_irrel),
            const.irrel_target_width,
            const.target_height,
            edgecolor=(*wheel.hex_to_rgb(partner_color, normalize=True),edge_alpha),
            facecolor="none",
            lw=border_width,
            clip_on=False,
            transform=ax.transData,
        ),
        "p2_relevant": mpl.patches.Rectangle(
            (x_rel+pert-translation, y_rel),
            const.rel_target_width,
            const.target_height,
            edgecolor=(*wheel.hex_to_rgb(partner_color, normalize=True),edge_alpha),
            facecolor="none",
            lw=border_width,
            clip_on=False,
            transform=ax.transData,
        ),
        "joint_relevant": mpl.patches.Rectangle(
            (x_rel+pert-translation, y_rel),
            const.rel_target_width,
            const.target_height,
            edgecolor=(*wheel.hex_to_rgb(partner_color, normalize=True),edge_alpha),
            facecolor="none",
            lw=border_width,
            clip_on=False,
            transform=ax.transData,
        ),
    }
    return p1_targets, p2_targets_copy

def aim2_legend(ax, labels, colors, xpos, ypos, long_width=0.1, short_width=0.015, height=0.4, fontsize=7, 
                target_pad=0.01, bbox_filled=False, lw=1.5):
    # This is for the legend for the onset time analysis where I don't plot the joint_irrelevant condition bc they don't respond
    if const.collapsed_condition_names[0] not in labels:
        skip_irrel = True
    else:
        skip_irrel = False
        
    if bbox_filled:
        facecolors=colors
        textcolors=['black']*len(colors)
    else:
        facecolors=['none']*len(colors)
        textcolors=colors
        
    text_bboxes = []
    text_objects = []
    for i, (label, facecolor, textcolor) in enumerate(zip(labels, facecolors, textcolors)):
        text = ax.text(
            xpos[i], ypos, label, color=textcolor, fontweight="bold", ha='left', va='bottom', multialignment="center",
            fontsize=fontsize,
            bbox=dict(facecolor=facecolor, edgecolor='none',boxstyle='round,pad=0.1')
        )
        text_objects.append(text)
        text_bboxes.append(text.get_window_extent().transformed(ax.transAxes.inverted()))
        
    xcorners = xpos + np.array([(b.x1 - b.x0)/2 for b in text_bboxes]) - 0.5*long_width
    ycorner = ypos - height - target_pad
    plot_legend_target_patches(ax, labels=labels, skip_irrel=skip_irrel, long_width=long_width, short_width=short_width, xcorners=xcorners, 
                              ycorner=ycorner, height=height, transform=ax.transAxes, lw=lw)
    return ax

def plot_legend_target_patches(ax, labels, skip_irrel, xcorners, ycorner, short_width=0.025, long_width=0.18, height=0.15, transform=None,
                               self_color = const.self_color, partner_color = const.partner_color, lw=1.5):
    if transform is None:
        transform = ax.transAxes
        
    rel_xcorners = xcorners + 0.5*long_width - 0.5*short_width
    if not skip_irrel:
        self_targets =  {
            "Self Irrelevant\nPartner Irrelevant": mpl.patches.Rectangle((xcorners[0], ycorner), long_width, height,
                edgecolor=self_color, facecolor=self_color,
                lw=0, clip_on=False, transform=transform, alpha=1,
            ),
            "Self Irrelevant\nPartner Relevant": mpl.patches.Rectangle((xcorners[1], ycorner), long_width, height,
                edgecolor=self_color, facecolor=self_color,
                lw=lw, clip_on=False, transform=transform, alpha=1,
            ),
            "Self Relevant\nPartner Irrelevant": mpl.patches.Rectangle((rel_xcorners[2], ycorner), short_width, height,
                edgecolor=self_color, facecolor=self_color,
                lw=0, clip_on=False, transform=transform, alpha=1,
            ),
            "Self Relevant\nPartner Relevant": mpl.patches.Rectangle((rel_xcorners[3], ycorner), short_width, height,
                edgecolor=self_color, facecolor=self_color,
                lw=0, clip_on=False, transform=transform, alpha=1,
            ),
        }
        partner_targets =  {
            "Self Irrelevant\nPartner Irrelevant": mpl.patches.Rectangle((xcorners[0], ycorner), long_width, height,
                edgecolor=partner_color, facecolor="none",
                lw=lw, clip_on=False, transform=transform, alpha=1,
            ),
            "Self Irrelevant\nPartner Relevant": mpl.patches.Rectangle((rel_xcorners[1], ycorner), short_width, height,
                edgecolor=partner_color, facecolor="none",
                lw=lw, clip_on=False, transform=transform, alpha=1,
            ),
            "Self Relevant\nPartner Irrelevant": mpl.patches.Rectangle((xcorners[2], ycorner), long_width, height,
                edgecolor=partner_color, facecolor="none",
                lw=lw, clip_on=False, transform=transform, alpha=1,
            ),
            "Self Relevant\nPartner Relevant": mpl.patches.Rectangle((rel_xcorners[3], ycorner), short_width, height,
                edgecolor=partner_color, facecolor="none",
                lw=lw, clip_on=False, transform=transform, alpha=1,
            ),
        }
    else:
        self_targets =  {
            "Self Irrelevant\nPartner Relevant": mpl.patches.Rectangle((xcorners[0], ycorner), long_width, height,
                edgecolor=self_color, facecolor=self_color,
                lw=lw, clip_on=False, transform=transform, alpha=1,
            ),
            "Self Relevant\nPartner Irrelevant": mpl.patches.Rectangle((rel_xcorners[1], ycorner), short_width, height,
                edgecolor=self_color, facecolor=self_color,
                lw=0, clip_on=False, transform=transform, alpha=1,
            ),
            "Self Relevant\nPartner Relevant": mpl.patches.Rectangle((rel_xcorners[2], ycorner), short_width, height,
                edgecolor=self_color, facecolor=self_color,
                lw=0, clip_on=False, transform=transform, alpha=1,
            ),
        }
        partner_targets =  {
            "Self Irrelevant\nPartner Relevant": mpl.patches.Rectangle((rel_xcorners[0], ycorner), short_width, height,
                edgecolor=partner_color, facecolor="none",
                lw=lw, clip_on=False, transform=transform, alpha=1,
            ),
            "Self Relevant\nPartner Irrelevant": mpl.patches.Rectangle((xcorners[1], ycorner), long_width, height,
                edgecolor=partner_color, facecolor="none",
                lw=lw, clip_on=False, transform=transform, alpha=1,
            ),
            "Self Relevant\nPartner Relevant": mpl.patches.Rectangle((rel_xcorners[2], ycorner), short_width, height,
                edgecolor=partner_color, facecolor="none",
                lw=lw, clip_on=False, transform=transform, alpha=1,
            ),
        }
        
    for i,k in enumerate(self_targets.keys()):
        ax.add_patch(self_targets[k])
        ax.add_patch(partner_targets[k])

def plot_ricatti_solution(timesteps, P1_sol, P2_sol, state_labels, state_mapping):
    fig = dv.AutoFigure('a;b', figsize=(5,6))
    ax1 = fig.axes['a']
    ax2 = fig.axes['b']
    xstate_ids = [state_mapping.get(key) for key in state_labels if key.find("x")!=-1]
    xstate_labels = [key for key in state_labels if key.find("x")!=-1]
    ystate_ids = [state_mapping.get(key) for key in state_labels if key.find("y")!=-1]
    ystate_labels = [key for key in state_labels if key.find("y")!=-1]

    for i in range(len(xstate_ids)):
        idx = xstate_ids[i]
        idy = ystate_ids[i]
        label = state_labels[i]
        
        ax1.plot(timesteps,P1_sol[:,idx,idx], label=f"P1 x {xstate_labels[i]}")
        ax1.plot(timesteps,P1_sol[:,idy,idy], label=f"P1 y {ystate_labels[i]}", ls='--')
        
        ax2.plot(timesteps,P2_sol[:,idx,idx], label=f"P2 x {xstate_labels[i]}")
        ax2.plot(timesteps,P2_sol[:,idy,idy], label=f"P2 y {ystate_labels[i]}", ls='--')
    
    for ax in [ax1,ax2]:
        ax.set_ylabel("P(t)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left")

    ax1.set_title("Player 1 Ricatti Solution")
    ax2.set_title("Player 2 Ricatti Solution")    
    
    
    fig.remove_figure_borders()
    return fig,ax1,ax2

def plot_all_feedback_gains(M, state_mapping, figsize=(40,10)):
    M = M[:,:len(state_mapping), :len(state_mapping)] # Handle augmentation

    if M.shape[1]>2 and figsize[1]<20:
        figsize = (40,40)
        
    fig,axes = plt.subplots(M.shape[1], M.shape[2], dpi=250, figsize=figsize)
    
    if M.shape[1] == 2:
        ylabs = ["x", "y"]
    else:
        ylabs = list(state_mapping.keys())
    titles = list(state_mapping.keys())
    c=-1
    for i in range(M.shape[1]):
        for j in range(M.shape[2]):
            c+=1    
            axes[i,j].plot(M[:,i,j])
            if i==0:
                axes[i,j].set_title(titles[j])
            
            if j==0:
                axes[i,j].set_ylabel(ylabs[i], fontsize=20)
    return fig,axes
        


def plot_feedback_gains(timesteps, F1, F2, state_labels, state_mapping, ax=None):
    if ax is None:
        fig = dv.AutoFigure('a;b', figsize=(5,6))
        ax1 = fig.axes['a']
        ax2 = fig.axes['b']
        
        for label in state_labels:
            if label.find("x")!=-1:
                ax1.plot(timesteps,F1[:,0,state_mapping[label]], label=f"P1 Fx {label}")
                ax2.plot(timesteps,F2[:,0,state_mapping[label]], label=f"P2 Fx {label}",ls="-.")
            else:
                ax1.plot(timesteps,F1[:,1,state_mapping[label]], label=f"P1 Fy {label}", ls='--')
                ax2.plot(timesteps,F2[:,1,state_mapping[label]], label=f"P2 Fy {label}", ls=':')
        
        for ax in [ax1,ax2]:
            
            ax.set_ylabel("F(t)")
            ax.set_xlabel("Time (s)")
            ax.legend(loc="upper left")

        ax1.set_title("Player 1 Feedback Gains")
        ax2.set_title("Player 2 Feedback Gains")    
        
        
        fig.remove_figure_borders()
        return fig,ax1,ax2
    
    else:
        for label in state_labels:
            if label.find("x")!=-1:
                ax.plot(timesteps,F1[:,0,state_mapping[label]], label=f"P1 Fx {label}")
                ax.plot(timesteps,F2[:,0,state_mapping[label]], label=f"P2 Fx {label}",ls="-.")
            else:
                ax.plot(timesteps,F1[:,1,state_mapping[label]], label=f"P1 Fy {label}", ls='--')
                ax.plot(timesteps,F2[:,1,state_mapping[label]], label=f"P2 Fy {label}", ls=':')
            
        ax.set_ylabel("F(t)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left")

        return ax    

def plot_control_signal(timesteps, u1, u2):
    fig = dv.AutoFigure('a;b', figsize=(5,6))
    ax1 = fig.axes['a']
    ax2 = fig.axes['b']
    ax1.plot(timesteps,u1[:,0], label=f"P1 u1x")
    ax1.plot(timesteps,u1[:,1], label=f"P1 u1y", ls='--')
    ax2.plot(timesteps,u2[:,0], label=f"P2 u2x")
    ax2.plot(timesteps,u2[:,1], label=f"P2 u2y", ls='--')
    
    for ax in [ax1,ax2]:
        
        ax.set_ylabel("u(t)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left")

    ax1.set_title("Player 1 Control Signal")
    ax2.set_title("Player 2 Control Signal")    
    
    
    fig.remove_figure_borders()
    return fig,ax1,ax2

def plot_xy_position(x, x_post, state_mapping, dual=True, muscle=False, separate_targets=False, ax=None):
    rhx_id,rhy_id = state_mapping["rhx"], state_mapping["rhy"]
    if separate_targets:
        rtx_id,rty_id = state_mapping["rtx"],state_mapping["rty"]
        ltx_id,lty_id = state_mapping["ltx"],state_mapping["lty"]
    else:
        ctx_id,cty_id = state_mapping["ctx"],state_mapping["cty"]
        
    if ax is None:
        fig= dv.AutoFigure("a")
        ax = fig.axes['a']
        
    ax.scatter(x[:, rhx_id],x[:, rhy_id],color=wheel.grey,marker="x",s=10,)  # Actual hand position scattered
    ax.plot(x_post[:, rhx_id], x_post[:, rhy_id], color=wheel.pink)  # Posterior estiamte of the cursor
    if dual:
        lhx_id,lhy_id = state_mapping["lhx"],state_mapping["lhy"]
        ccx_id,ccy_id = state_mapping["ccx"],state_mapping["ccy"]
        ax.scatter(x[:, lhx_id],x[:, lhy_id],color=wheel.grey,marker="x",s=10,)  # Actual hand position scattered
        ax.plot(x_post[:, lhx_id], x_post[:, lhy_id], color=wheel.rak_blue)  # Posterior estiamte of the cursor

        ax.scatter(x[:, ccx_id],x[:, ccy_id],color=wheel.grey,marker="x",s=10,)  # Posterior estiamte of the center cursor
        ax.plot(x_post[:, ccx_id],x_post[:, ccy_id],color=wheel.sunflower,)  # Posterior estiamte of the center cursor
    
    if separate_targets:
        ax.scatter(x[-1,rtx_id], x[-1,rty_id], c=wheel.pink)
        ax.scatter(x[-1,ltx_id], x[-1,lty_id], facecolor="none", edgecolor=wheel.rak_blue)
    else:
        ax.scatter(x[-1,ctx_id], x[-1,cty_id], c=wheel.sunflower)

    ax.set_xlabel("X-Position")
    ax.set_ylabel("Y-Position")
    dv.Custom_Legend(
        ax,
        labels=["Right Hand", "Left Hand"],
        colors=[wheel.pink, wheel.rak_blue],
        fontsize=8,
    )
    
    dv.Custom_Legend(ax, labels=["P1", "P2"], colors=[wheel.rak_red, wheel.rak_blue], loc = (0.1,1), fontsize=12)
    if ax is None:
        return fig
    else:
        return ax

def plot_states_over_time(x, state_labels, state_mapping, linestyles, ax = None, colors=None):
    if ax is None:
        fig = dv.AutoFigure('a')
        ax = fig.axes['a']
    if colors is None:
        colors = [wheel.pink, wheel.rak_blue, wheel.red, wheel.blue, 
                  wheel.sunflower, wheel.lighten_color(wheel.sunflower,1.25)]
    # Plot x,y traces for lh, rh, and cc
    state_ids = [state_mapping.get(k) for k in state_labels]
    for idx, label,ls,color in zip(state_ids,state_labels,linestyles,colors):
        ax.plot(x[:,idx], label=label, ls=ls, c=color)
    
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("State")
    ax.legend()
    
    if ax is None:
        return fig
    else:
        return ax

def create_trajectory_animation(
    trial_type,
    rhx, rhy, lhx, lhy, ccx, ccy, 
    p1_applied_force, p2_applied_force, timepoints, probe_timepoints, 
    rtx=None, rty=None, ltx=None, lty=None,
    save_path=None, save_name=None, text_color='red',
    probe_onset = 0, alpha=0.0, partner_knowledge=False, 
    target_type="joint_irrelevant", pause_time=0, perturbation_type="Cursor Jump",
    save = False,
):
    """Creates an animation of model trajectories for a given condition
    
    Args:
        regular_df (pl.DataFrame): DataFrame containing model trajectories
        save_path (Path): Path to save the animation
        alpha (float): Model alpha parameter
        partner_knowledge (bool): Whether the model has knowledge of partner's target
        condition (str): Condition name to filter by
        trial_number (int): Trial number to animate
    """
    # Set up the figure
    if trial_type == "probe":
        fig,axes = plt.subplots(1,2, figsize=(8,3.2))
        ax1,ax2 = axes 
    else:
        fig,ax1 = plt.subplots(figsize=(4, 4))
    
    # Initialize line objects for trajectories (thin lines)
    rh_line, = ax1.plot([], [], '-', color=const.self_color, linewidth=2, alpha=1, zorder=10)
    lh_line, = ax1.plot([], [], '-', color=const.partner_color, linewidth=2, alpha=1, zorder=10)
    cc_line, = ax1.plot([], [], '-', color=const.cursor_colors['cc'], linewidth=2, alpha=1, zorder=10)
    
    # Initialize marker objects for current positions (larger circles)
    rh_marker, = ax1.plot([], [], 'o', color=const.self_color, markersize=5, zorder=10)
    lh_marker, = ax1.plot([], [], 'o', color=const.partner_color, markersize=5, zorder=10)
    cc_marker, = ax1.plot([], [], 'o', color=const.cursor_colors['cc'], markersize=4, zorder=10)
    p1_target, = ax1.plot([], [], 's', color=const.self_color, lw=0, markersize=10, zorder=8)
    p2_target, = ax1.plot([], [], 's', markerfacecolor='none', markeredgecolor=const.partner_color, lw=0, markeredgewidth=2, markersize=10, zorder=9)
    if rtx is None:
        p1_targets, p2_targets = get_target_patches(ax1, target_jump=False, 
                                                       self_color=const.self_color, partner_color=const.partner_color, 
                                                       border_width=1.)
        patch1 = []
        patch2 = []
    # Create jump spot
    if trial_type != "regular":
        ax1.text(0.48, 0.25*(0.25) + rhy[0], "Cursor\nJump", ha='center',va='center', fontsize=6, color=wheel.grey, rotation=0)
        ax1.axhline(0.25*(0.25) + rhy[0], xmin=0.33, xmax=0.67, color='gray', linestyle='--', linewidth=0.8)
    
    # Create vline and force channel arrows
    if trial_type == "probe":
        ax1.plot((rhx[0],rhx[0]), (rhy[0], rhy[0]+0.25), color='gray', linestyle='-', linewidth=1)
        ax1.plot((lhx[0],lhx[0]), (lhy[0], lhy[0]+0.25), color='gray', linestyle='-', linewidth=1)
        arrow_length = 0.03
        arrow_style = dict(head_width=0.005, head_length=0.005, color=wheel.white, lw=1.5, length_includes_head=True, 
                        alpha=1)
        for ypos in np.linspace(0.07, 0.28, 6):
            # RIght hand arrows
            ax1.arrow(rhx[0]-arrow_length-0.008, ypos, arrow_length, 0, **arrow_style)
            ax1.arrow(rhx[0]+arrow_length+0.008, ypos, -arrow_length, 0, **arrow_style)
            # Left hand arrows
            ax1.arrow(lhx[0]-arrow_length-0.008, ypos, arrow_length, 0, **arrow_style)
            ax1.arrow(lhx[0]+arrow_length+0.008, ypos, -arrow_length, 0, **arrow_style)
        
        fps = 15
    else:
        fps = 45
    
    
    ax1.set_xlim(lhx[0]-0.1,rhx[0]+0.1)
    ax1.set_ylim(0.0, 0.35)
    ax1.set_aspect('equal')
    ax1.set_axis_off()
    dv.Custom_Legend(ax1, 
                     colors=[const.partner_color, const.cursor_colors['cc'], const.self_color],
                     labels=["Partner", "Center Cursor", "Self"],ncol=3, loc="upper center", columnspacing=4)
    ax1.set_title(f'{const.target_types_to_collapsed_conditions[target_type]}', fontsize=9, c=text_color)
    # ax1.set_position([-0.1, -0.02, 0.8, 0.8])
    
    #* Initialize ax2 for the feedback response
    if trial_type == "probe":
        ylim = (-0.2,2.5)
        xticks = np.arange(0,401,100)

        p1_force, = ax2.plot([], [], '-', color=text_color, linewidth=2, alpha=1)
        p2_force, = ax2.plot([], [], '-', color=const.partner_color, linewidth=2, alpha=1)
        
        ax2.axvline(x=0, ls="--", c= wheel.grey,lw=0.75)
        ax2.text(0,ylim[1]/2, perturbation_type, rotation=90, va='center', ha='right', color=wheel.grey, fontsize=6)
        
        # Plot involuntary region
        # ax.fill_betweenx(np.arange(ylim[0],ylim[1],0.01), 180, 230, facecolor = wheel.lighten_color(wheel.light_grey,0.75), alpha = 0.1)
        ax2.text(205,ylim[-1],"Involuntary", rotation=90, va='top', ha='center', color=wheel.grey, fontsize=6)
        ax2.axvline(x=180, color='grey', ls='-', lw=0.5, zorder=-10)
        ax2.axvline(x=230, color='grey', ls='-', lw=0.5, zorder=-10)

        ax2.set_xticks(xticks)
        
        ax2.set_xlabel("Time From Cursor Jump (ms)")
        ax2.set_ylabel("Visuomotor Feedback Response [N]")
        ax2.set_xlim(-25, 400)
        ax2.set_ylim(ylim)
        # dv.Custom_Legend(ax2, 
        #              colors=[const.partner_color, const.self_color],
        #              labels=["Partner", "Self"],ncol=1, loc="upper right", columnspacing=4)
        global index 
        index = -1

    def init():
        # Initialize both lines and markers empty
        rh_line.set_data([], [])
        lh_line.set_data([], [])
        cc_line.set_data([], [])
        rh_marker.set_data([], [])
        lh_marker.set_data([], [])
        cc_marker.set_data([], [])
        if rtx is not None:
            p1_target.set_data([], [])
            p2_target.set_data([],[])
        else:
            ax1.add_patch(p1_targets[target_type])
            ax1.add_patch(p2_targets[target_type])

            
            
        

        if trial_type == "probe":
            p1_force.set_data([], [])
            p2_force.set_data([], [])
            return_tuple = (rh_line, lh_line, cc_line, rh_marker, lh_marker, cc_marker, p1_force)
        else:
            return_tuple = (rh_line, lh_line, cc_line, rh_marker, lh_marker, cc_marker)
        return return_tuple
    
    def animate(frame):
        global index
        # Update trajectory lines (show full path up to current frame)
        rh_line.set_data(rhx[:frame], rhy[:frame])
        lh_line.set_data(lhx[:frame], lhy[:frame])
        cc_line.set_data(ccx[:frame], ccy[:frame])
        
        # Update current position markers (show only at current frame)
        rh_marker.set_data([rhx[frame]], [rhy[frame]])
        lh_marker.set_data([lhx[frame]], [lhy[frame]])
        cc_marker.set_data([ccx[frame]], [ccy[frame]])
        
        # Update Force applied
        if trial_type == "probe":
            if timepoints[frame]*1000 >= probe_onset-50:
                index+=1
                p1_force.set_data(probe_timepoints[:index], p1_applied_force[:index])
                p2_force.set_data(probe_timepoints[:index], p2_applied_force[:index])

        if rtx is None:
            patch1.append(p1_targets[target_type])
            patch2.append(p2_targets[target_type])
        else:
            p1_target.set_data(rtx[frame], rty[frame])
            p2_target.set_data(ltx[frame], lty[frame])
        
        if trial_type == "probe":
            return_tuple = (rh_line, lh_line, cc_line, rh_marker, lh_marker, cc_marker, p1_force)
        else:
            return_tuple = (rh_line, lh_line, cc_line, rh_marker, lh_marker, cc_marker)
        
        return return_tuple
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(rhx), interval=30, blit=True)
    
    # Save animation
    # NOTE Interval must be divisible by fps for ffmpeg to work properly, JK interval doesn't matter
    if save:
        anim.save(save_path / f'{save_name}_animation_alpha_{alpha}_pk_{partner_knowledge}_condition_{target_type}.gif', 
                    writer='ffmpeg', fps=fps)
    # plt.close()