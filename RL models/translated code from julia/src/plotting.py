import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from matplotlib.lines import Line2D
from src.ToPlanOrNotToPlan import *
from walls import *

# Set some reasonable plotting defaults
plt.rcParams.update({
    'font.size': 16,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def plot_progress(rews, vals, fname="figs/progress.png"):
    plt.figure(figsize=(6, 2.5))
    axs = [121, 122]
    data = [rews, vals]
    ts = np.arange(1, len(rews) + 1)
    labs = ["reward", "prediction"]

    for i in range(2):
        plt.subplot(axs[i])
        plt.plot(ts, data[i], 'k-')
        plt.xlabel("epochs")
        plt.ylabel(labs[i])
        plt.title(labs[i])

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def arena_lines(ps, wall_loc, Larena, rew=True, col="k", rew_col="k", lw_arena=1., col_arena=np.ones(3) * 0.3, lw_wall=10):
    Nstates = Larena**2

    for i in range(Larena + 1):
        plt.axvline(i + 0.5, color=col_arena, lw=lw_arena)
        plt.axhline(i + 0.5, color=col_arena, lw=lw_arena)

    if rew:
        rew_loc = state_from_onehot(Larena, ps)
        plt.scatter(rew_loc[0], rew_loc[1], c=rew_col, marker="*", s=350, zorder=50)  # Reward location

    for s in range(1, Nstates + 1):
        for i in range(1, 5):
            if wall_loc[s-1, i-1]:
                state = state_from_loc(Larena, s)
                if i == 1:  # Wall to the right
                    z1, z2 = state + [0.5, 0.5], state + [0.5, -0.5]
                elif i == 2:  # Wall to the left
                    z1, z2 = state + [-0.5, 0.5], state + [-0.5, -0.5]
                elif i == 3:  # Wall above
                    z1, z2 = state + [0.5, 0.5], state + [-0.5, 0.5]
                elif i == 4:  # Wall below
                    z1, z2 = state + [0.5, -0.5], state + [-0.5, -0.5]
                plt.plot([z1[0], z2[0]], [z1[1], z2[1]], color=col, linestyle="-", linewidth=lw_wall)

    plt.xlim(0.49, Larena + 0.52)
    plt.ylim(0.48, Larena + 0.51)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

def plot_arena(ps, wall_loc, Larena, ind=1):
    ps = ps[:, ind - 1]
    wall_loc = wall_loc[:, :, ind - 1]
    plt.figure(figsize=(6, 6))
    arena_lines(ps, wall_loc, Larena)
    plt.savefig("figs/wall/test_arena.png", bbox_inches="tight")
    plt.close()

def plot_rollout(state, rollout, wall, Larena):
    col = [0.5, 0.8, 0.5] if rollout[-1] else [0.5, 0.5, 0.8]
    rollout = np.array(rollout, dtype=int)

    for a in rollout[:-1]:
        if a > 0:
            if wall[state_ind_from_state(Larena, state)[0], a] > 0:
                new_state = state
            else:
                direction = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
                new_state = state + direction[a-1]

            new_state = (new_state + Larena - 1) % Larena + 1
            x1, x2 = min(state[0], new_state[0]), max(state[0], new_state[0])
            y1, y2 = min(state[1], new_state[1]), max(state[1], new_state[1])

            lw = 5
            if x2 - x1 > 1.5:
                plt.plot([x1, x1-0.5], [y1, y2], linestyle="-", color=col, linewidth=lw)
                plt.plot([x2, x2+0.5], [y1, y2], linestyle="-", color=col, linewidth=lw)
            elif y2 - y1 > 1.5:
                plt.plot([x1, x2], [y1, y1-0.5], linestyle="-", color=col, linewidth=lw)
                plt.plot([x1, x2], [y2, y2+0.5], linestyle="-", color=col, linewidth=lw)
            else:
                plt.plot([x1, x2], [y1, y2], linestyle="-", color=col, linewidth=lw)

            state = new_state

def plot_weiji_gif(ps, wall_loc, states, as_, rews, Larena, RTs, fname, Tplot=10, res=60, minframe=3, figsize=(4, 4), rewT=400, fix_first_RT=True, first_RT=500, plot_rollouts=False, rollout_time=120, rollouts=[], dpi=80, plot_rollout_frames=False):
    os.makedirs(f"{fname}_temp", exist_ok=True)
    Tplot = Tplot * 1e3 / res  # Now in units of frames

    if fix_first_RT:
        RTs[:, 0] = first_RT

    for batch in range(1, ps.shape[1] + 1):
        bstr = f"{batch:02d}"
        rew_loc = state_from_onehot(Larena, ps[:, batch - 1])
        Rtot = 0
        t = 0  # Real time
        anum = 0  # Number of actions
        rew_col = "lightgrey"

        while (anum < np.sum(as_[batch - 1, :] > 0.5)) and (t < Tplot):
            anum += 1

            astr = f"{anum:03d}"
            print(f"{bstr} {astr}")

            RT = RTs[batch - 1, anum - 1]  # Reaction time for this action
            nframe = max(round(RT / res), minframe)  # Number of frames to plot (at least three)
            rewframes = 0  # No reward frames

            if rews[batch - 1, anum - 1] > 0.5:
                rewframes = round(rewT / res)  # Add a singular frame at reward

            if (anum > 1.5) and (rews[batch - 1, anum - 2] > 0.5):
                rew_col = "k"  # Show that we've found the reward

            R_increased = False  # Have we increased R for this action
            frames = np.arange(minframe - nframe + 1, minframe + rewframes)
            frolls = {f: 0 for f in frames}  # Dictionary pointing to rollout; 0 is None

            if plot_rollouts or plot_rollout_frames:  # Either plot rollouts or the corresponding frames
                nroll = np.sum(rollouts[batch - 1, anum - 1, 0, :] > 0.5)  # How many rollouts?
                print(f"rolls: {nroll}")
                f_per_roll = round(rollout_time / res)  # Frames per rollout
                frames = np.arange(min(frames[0], -nroll * f_per_roll + 1), frames[-1] + 1)  # Make sure there is enough frames for plotting rollouts
                frolls = {f: 0 for f in frames}  # Dictionary pointing to rollout; 0 is None

                for roll in range(1, nroll + 1):
                    new_rolls = np.arange(-(f_per_roll * roll - 1), -(f_per_roll * (roll - 1)))
                    frac = 0.5 if nroll == 1 else (roll - 1) / (nroll - 1)  # [0, 1]
                    new_roll_1 = round(frames[0] * frac - (f_per_roll - 1) * (1 - frac))
                    new_rolls = np.arange(new_roll_1, new_roll_1 + f_per_roll)
                    for r in new_rolls:
                        frolls[r] = nroll - roll + 1

            for f in frames:
                state = states[:, batch - 1, anum - 1]
                fstr = f"{f - frames[0]:03d}"

                frac = min(max(0, (f - 1) / minframe), 1)
                plt.figure(figsize=figsize)

                arena_lines(ps[:, batch - 1], wall_loc[:, :, batch - 1], Larena, col="k", rew_col=rew_col)

                col = "b"
                if (rewframes > 0) and (frac >= 1):
                    col = "g"  # Colour green when at reward
                    if not R_increased:
                        Rtot += 1
                        R_increased = True

                if plot_rollouts:  # Plot the rollout
                    if frolls[f] > 0.5:
                        plot_rollout(state, rollouts[batch - 1, anum - 1, :, frolls[f]], wall_loc[:, :, batch - 1], Larena)

                a = as_[batch - 1, anum - 1]  # Higher resolution
                state += frac * np.array([int(a == 1) - int(a == 2), int(a == 3) - int(a == 4)])  # Move towards next state
                state = (state + Larena - 0.5) % Larena + 0.5
                plt.scatter(state[0], state[1], marker="o", color=col, s=200, zorder=100)

                tstr = f"{t:03d}"
                t += 1
                realt = t * res * 1e-3
                print(f"{anum} {f} {round(frac, 2)} {RT} {round(realt, 1)}")
                plt.title(f"t = {round(realt, 1)} (R = {Rtot})")

                astr = f"{anum:02d}"
                if t <= Tplot:
                    plt.savefig(f"{fname}_temp/temp{bstr}_{tstr}_{fstr}_{astr}.png", bbox_inches="tight", dpi=dpi)
                plt.close()

    # Combine PNGs to GIF
    os.system(f"convert -delay {int(round(res / 10))} {fname}_temp/temp*.png {fname}.gif")
    os.system(f"rm -r {fname}_temp")
