# %%
## Imports

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors

import pickle

from scipy.signal import correlate
from scipy.stats import mannwhitneyu
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


# %%
## Select participant

participant = 'T16'
# participant = 'T11'

# define save path for plots
save_plot_filepath = './plots/fig5/'
if not os.path.exists(save_plot_filepath):
    os.makedirs(save_plot_filepath)


# %%
## Participant-specific parameters

# bin width in seconds, and the same bin width in ms
if participant == 'T16':
    BIN_WIDTH = 0.02  # seconds
elif participant == 'T11':
    BIN_WIDTH = 0.01  # seconds
TS = int(1000*BIN_WIDTH)

# decoding parameters: smoothing factor, move-state threshold, and prep window duration
if participant == 'T16':
    alpha = 0.889
    threshold_on = 0.9
    prep_window_len = 800
elif participant == 'T11':
    alpha = 0.933
    threshold_on = 0.98
    prep_window_len = 200


# %%
## Load continuous data

# load continuous data, trial info, clock time, and the kept (active) channel indices
with h5py.File(f'./data/fig5_{participant}_continuous.h5', 'r') as cont_h5:
    cont_data = {k: cont_h5['data'][k][()] for k in cont_h5['data'] if k != 'clock_time'}
    cont_trial_info = {k: cont_h5['trial_info'][k][()] for k in cont_h5['trial_info']}
    cont_time = cont_h5['data']['clock_time'][()]
    keep_chans = cont_h5['keep_chans'][()]

# field for continuous neural data
neural_dec_field = 'neural_data_norm'


# %%
## Extract and reprocess continuous data for plotting

# get and smooth neural activity (exponential filter)
cont_neural_data = cont_data[neural_dec_field]
cont_neural_data_smooth = np.zeros(cont_neural_data.shape)
for i in range(1, cont_neural_data.shape[0]):
    cont_neural_data_smooth[i] = cont_neural_data_smooth[i-1] * alpha + (1-alpha) * cont_neural_data[i]

# get decoded move state and kinematics
move_pred = cont_data['dec_move_state'][:, 0]
cont_kinematics = cont_data['control_samples']

# get EMG data if available
EMG_LAG_MS = 40  # EMG recording delay (compensate by shifting signal earlier)
if 'emg_data' in cont_data:
    cont_emg = cont_data['emg_data']

# get trial event times (converted to seconds)
t_cont = cont_time.astype(float) / 1e3
start_times = cont_trial_info['start_time'].astype(float) / 1e3
go_times = cont_trial_info['go_cue_time'].astype(float) / 1e3
# get decoded onset times (absolute idx; -1 = no crossing)
dec_onsets = {i: int(ci) for i, ci in
             enumerate(cont_trial_info['dec_movement_onset_time']) if ci >= 0}

# if no_go_bool field is present, use it to identify no-go trials;
# otherwise, assume all trials are go trials
if 'no_go_bool' in cont_trial_info:
    trial_info_no_go_trial = cont_trial_info['no_go_bool']
else:
    trial_info_no_go_trial = np.zeros_like(start_times, dtype=int)


# %%
## Plot windows of continuous data

def draw_lines(ax):
    """
    Draw vertical lines for trial start times (yellow) and go cue times (green) on the given axis.
    """
    for start_time in start_times:
        ax.axvline(x=start_time, color='y', linestyle='-', linewidth=5)
    for i, go_time in enumerate(go_times):
        if trial_info_no_go_trial[i] == 0:
            ax.axvline(go_time, color='g', linestyle='-', linewidth=5)

# compute the raster color limits (1st/99th percentiles)
neural_data_percentiles = np.percentile(cont_neural_data_smooth.flatten(), [1, 99])

# select start time of windows to plot: extended window w/ EMG, shorter main window,
# or zoomed-in window
if participant == 'T16':
    time_offsets = {'w_emg': 54, 'main': 53, 'zoom': 63.387}
else:
    time_offsets = {'w_emg': 10, 'main': 10, 'zoom': 10}

# loop over figures: continuous data (main), extended continuous data w/ EMG
for name, figsize, time_offset, time_window, with_emg in [
        ('continuous_data',       (15, 7),  time_offsets['main'],  15,   False),
        ('continuous_data_w_emg', (22, 10), time_offsets['w_emg'], 21.5, True),
    ]:

    # plot EMG only in the extended figure, and only if EMG is present
    plot_emg = with_emg and 'emg_data' in cont_data
    height_ratios = [1.5, 0.75, 1.5] + ([1.5] if plot_emg else [])

    _, axs = plt.subplots(len(height_ratios), 1, figsize=figsize, facecolor='w',
                          sharex=True, height_ratios=height_ratios)

    # 1. plot neural activity
    ax = axs[0]

    ax.imshow(cont_neural_data_smooth[:,keep_chans].T,
            aspect='auto',
            extent=[t_cont[0], t_cont[-1], 0, keep_chans.shape[0]],
            cmap='magma', vmin=neural_data_percentiles[0], vmax=neural_data_percentiles[-1])

    # highlight prep activity windows and decoded onsets
    for onset in dec_onsets.values():
        ax.axvspan(t_cont[onset-int(prep_window_len/TS)],
                    t_cont[onset], color='y', alpha=0.25)
        ax.axvline(t_cont[onset], color=(40/255,170/255,225/255), linestyle='-', linewidth=4, alpha=1)

    # draw trial events
    draw_lines(ax)

    # misc. subplot settings
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('Neural\nfeatures', fontsize=18)

    # 2. plot move state
    ax = axs[1]

    ax.plot(t_cont, move_pred, c=(0.3,0.3,0.3), alpha=1, linewidth=3, zorder=-100)

    # highlight prep activity windows and decoded onsets
    for onset in dec_onsets.values():
        ax.scatter(t_cont[onset], threshold_on, c=(40/255,170/255,225/255), alpha=1, s=100, marker='o')
        ax.axvspan(t_cont[onset-int(prep_window_len/TS)],
                    t_cont[onset], color='y', alpha=0.25)
        ax.axvline(t_cont[onset], color=(40/255,170/255,225/255), linestyle='-', linewidth=4, alpha=1)

    # draw probability threshold line
    ax.axhline(threshold_on, color='k', alpha=0.75, linestyle='--', linewidth=2, zorder=-200)

    # draw trial events
    draw_lines(ax)

    # misc. subplot settings
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylabel('Movement\nstate prob.', fontsize=18)
    ax.set_ylim(-0.01, 1.01)

    # 3. plot kinematics
    ax = axs[2]

    ax.plot(t_cont, cont_kinematics[:,0], c=(0.8,0,0), alpha=1, linewidth=3, label='x vel.')
    ax.plot(t_cont, cont_kinematics[:,1], c=(0.4,0,0), alpha=1, linewidth=3, label='y vel.')

    # highlight prep activity windows and decoded onsets
    for onset in dec_onsets.values():
        ax.axvspan(t_cont[onset-int(prep_window_len/TS)],
                    t_cont[onset], color='y', alpha=0.25)
        ax.axvline(t_cont[onset], color=(40/255,170/255,225/255), linestyle='-', linewidth=4, alpha=1)

    # draw trial events
    draw_lines(ax)

    # misc. subplot settings
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=14, loc='upper left')
    ax.set_ylabel('Cursor\nvelocity', fontsize=18)

    # 4. plot EMG activity if available
    if plot_emg:

        ax = axs[3]

        ax.plot(t_cont - EMG_LAG_MS / 1000, cont_emg[:,0], c='b', alpha=0.75, linewidth=3)

        # draw trial events
        draw_lines(ax)
        for onset in dec_onsets.values():
            ax.axvline(t_cont[onset], color=(40/255,170/255,225/255), linestyle='-', linewidth=4)

        # misc. subplot settings
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('EMG activity', fontsize=18)

    # only plot the defined window
    axs[0].set_xlim(time_offset, time_offset+time_window)

    # time scale bar
    axs[2].plot([time_offset+0.25, time_offset+0.25+1], [-20,-20],
            c='k', alpha=1, linewidth=8)

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig5_{participant}_{name}.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
## Plot zoomed-in windows of continuous data

time_offset = time_offsets['zoom']
time_window = 3

for name, signal, figsize in [('zoomed_neural',     'neural',     (5, 2.14)),
                              ('zoomed_movestate',  'move_state', (5, 1.43)),
                              ('zoomed_kinematics', 'kinematics', (4, 1))]:

    plt.figure(figsize=figsize, facecolor='w')

    # plot neural activity
    if signal == 'neural':
        plt.imshow(cont_neural_data_smooth[:,keep_chans].T,
                aspect='auto',
                extent=[t_cont[0], t_cont[-1], 0, keep_chans.shape[0]],
                cmap='magma', vmin=neural_data_percentiles[0], vmax=neural_data_percentiles[-1])
        # highlight prep activity window
        for onset in dec_onsets.values():
            plt.axvspan(t_cont[onset-int(prep_window_len/TS)],
                        t_cont[onset], color='y', alpha=0.3)

    # plot move state
    elif signal == 'move_state':
        plt.plot(t_cont, move_pred, c=(0.2,0.2,0.2), linewidth=2)
        # draw threshold line
        plt.axhline(threshold_on, color='k', alpha=0.75, linestyle='--', linewidth=1.5)
        plt.ylim(-0.05, 1.05)

    # plot kinematics
    elif signal == 'kinematics':
        plt.plot(t_cont, cont_kinematics[:,0], c=(0.8,0,0), alpha=0.75, linewidth=3)
        plt.plot(t_cont, cont_kinematics[:,1], c=(0.4,0,0), alpha=0.75, linewidth=3)

    # misc. plot settings
    plt.xlim(time_offset, time_offset+time_window)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig5_{participant}_{name}.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
## EMG vs. decoded movement-state cross-correlation

# only compute if EMG data is available
if 'emg_data' in cont_data:

    # use first EMG channel
    emg_env     = cont_emg[:,0]

    # correct for EMG recording lag (EMG_LAG_MS)
    lag_bins   = int(round(EMG_LAG_MS / TS))
    emg_env_corr = np.empty_like(emg_env)
    emg_env_corr[:-lag_bins] = emg_env[lag_bins:]
    emg_env_corr[-lag_bins:] = emg_env[-1]
    emg_env = emg_env_corr

    # time axis for move_pred/EMG aligned to decoded onset
    PRE_BINS  = int(1000 / TS)
    POST_BINS = int(1000 / TS)
    t_axis_ms = (np.arange(PRE_BINS + POST_BINS) - PRE_BINS) * TS

    # z-score data so that correlation at lag 0 equals Pearson r
    move_pred_z   = (move_pred - move_pred.mean()) / (move_pred.std() + 1e-8)
    emg_z  = (emg_env   - emg_env.mean())   / (emg_env.std()   + 1e-8)

    # compute cross-correlation (normalized by length of move_pred_z)
    xcorr  = correlate(emg_z, move_pred_z, mode='full') / len(move_pred_z)
    center = len(xcorr) // 2
    # trim to ±1000 ms (or 1000/TS bins) around lag=0 for plotting
    MAX_LAG_BINS = int(1000 / TS)
    xcorr_trim = xcorr[center - MAX_LAG_BINS : center + MAX_LAG_BINS + 1]
    lags_ms    = np.arange(-MAX_LAG_BINS, MAX_LAG_BINS + 1) * TS

    # compute peak cross-correlation and lag
    peak_lag_ms = lags_ms[np.argmax(xcorr_trim)]
    peak_r      = np.max(xcorr_trim)
    print(f'Peak cross-correlation: r={peak_r:.3f} at lag={peak_lag_ms:.0f} ms '
          '(positive = move_pred leads EMG)')

    # plot results
    fig, axs = plt.subplots(1, 2, figsize=(22, 6), facecolor='w')

    # ±1000 ms x range for both subplots
    XLIM = (lags_ms[0], lags_ms[-1])

    # left subplot: trial-averaged overlay aligned to decoded onset, dual y-axes (move_pred and EMG)
    ax  = axs[0]
    ax2 = ax.twinx()

    # generate arrays of trialized move_pred and EMG aligned to decoded onset
    move_pred_trials = []
    emg_trials  = []
    for ic in dec_onsets.values():
        i0, i1 = ic - PRE_BINS, ic + POST_BINS
        if i0 < 0 or i1 > len(move_pred):
            continue
        move_pred_trials.append(move_pred[i0:i1])
        emg_trials.append(emg_env[i0:i1])
    move_arr = np.array(move_pred_trials)
    emg_arr  = np.array(emg_trials)

    move_mean = move_arr.mean(axis=0)
    move_sem  = move_arr.std(axis=0) / np.sqrt(move_arr.shape[0])
    emg_mean  = emg_arr.mean(axis=0)
    emg_sem   = emg_arr.std(axis=0)  / np.sqrt(emg_arr.shape[0])

    # plot move state probability with SEM shading
    ax.fill_between(t_axis_ms, move_mean - move_sem, move_mean + move_sem,
                    alpha=0.25, color=(0.3, 0.3, 0.3))
    ax.plot(t_axis_ms, move_mean, color=(0.3, 0.3, 0.3), linewidth=4,
            label='Move state prob.')
    ax.axhline(threshold_on, color='k', linewidth=2.5, linestyle=':', alpha=0.5)

    # plot EMG activity with SEM shading
    ax2.fill_between(t_axis_ms, emg_mean - emg_sem, emg_mean + emg_sem,
                     alpha=0.25, color='b')
    ax2.plot(t_axis_ms, emg_mean, color='b', linewidth=4,
             label='Average EMG envelope', zorder=100)

    # draw vertical line at decoded onset (t=0)
    ax.axvline(0, color=(40/255, 170/255, 225/255), linewidth=6,
               linestyle='-', label='Decoded onset', zorder=100)

    # misc. subplot settings
    ax.set_xlabel('Time relative to decoded onset (ms)', fontsize=24)
    ax.set_ylabel('Average predicted\nmovement state probability', fontsize=24, color=(0.3, 0.3, 0.3))
    ax2.set_ylabel('Average EMG activity (AU)', fontsize=24, color='b')
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelcolor=(0.3, 0.3, 0.3), labelsize=18)
    ax2.tick_params(axis='y', labelcolor='b', labelsize=18)
    ax.set_xlim(*XLIM)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # right subplot: cross-correlation vs. lag
    ax = axs[1]

    # plot cross-correlation with peak lag indicated
    ax.plot(lags_ms, xcorr_trim, color=(0.2, 0.2, 0.2), linewidth=4)
    ax.axvline(peak_lag_ms, color='r', linewidth=2.5, linestyle='--',
               label=f'Peak at {peak_lag_ms:.0f} ms\n(r={peak_r:.2f})')

    # misc. subplot settings
    ax.set_xlabel('EMG − movement state probability lag (ms)', fontsize=24)
    ax.set_ylabel('Normalized cross-correlation', fontsize=24)
    ax.tick_params(labelsize=18)
    ax.set_xlim(*XLIM)
    ax.set_ylim(top=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=20, loc='lower center', bbox_to_anchor=(0.7, 0.02), framealpha=1)

    # adjust layout
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.45)

    # save pdf
    savepath = os.path.join(save_plot_filepath,
                            f'fig5_{participant}_emg_movestate_xcorr.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
## Compute decoded onset histogram and cumulative distribution

# load trialized data dict
with h5py.File(f'./data/fig5_{participant}_start.h5', 'r') as f:
    eval_trials = {k: f[k][()] for k in f}

# get trial-start index
start_bin = int(np.where(eval_trials['t_data'] == 0)[0][0])

# get trial event bins (go cue, decoded onset, trial end, start_bins)
go_bins        = eval_trials['go_bin_all'].astype(int)
dec_onset_bins = eval_trials['dec_move_onset_bin_all'].astype(int)
trial_end_bins = eval_trials['end_bin_all'].astype(int)
start_bins     = np.full_like(dec_onset_bins, start_bin)

# get no-go trial bools (1 = no-go, 0 = go)
no_go          = eval_trials['no_go_bool_all'].astype(int)

# define reference bins (go cue for T16, trial start for T11)
ref_bins = go_bins if participant == 'T16' else start_bins

# compute decoded onset class for each trial (0=good, 1=early, 2=wrong, 3=missing, 4=catch)
dec_onset_class = np.full(len(dec_onset_bins), 3, dtype=int)    # MISSING (default, no onset detected)
dec_onset_class[(no_go == 1) & (dec_onset_bins != -1)] = 2      # WRONG   (catch, onset detected)
dec_onset_class[(no_go == 1) & (dec_onset_bins == -1)] = 4      # CATCH   (catch, no onset detected)
dec_onset_class[(no_go == 0) & (dec_onset_bins != -1) &
               (dec_onset_bins >= ref_bins)] = 0                # GOOD    (non-catch, onset at/after ref)
dec_onset_class[(no_go == 0) & (dec_onset_bins != -1) &
               (dec_onset_bins <  ref_bins)] = 1                # EARLY   (non-catch, onset before ref)

good_crossings    = list(np.where(dec_onset_class == 0)[0])
early_crossings   = list(np.where(dec_onset_class == 1)[0])
wrong_crossings   = list(np.where(dec_onset_class == 2)[0])
missing_crossings = list(np.where(dec_onset_class == 3)[0])
crossing_trials = len(good_crossings) + len(early_crossings) + len(wrong_crossings) + len(missing_crossings)

print(f'missing crossings: {missing_crossings}')
print(f'wrong crossings: {wrong_crossings}')
print(f'early crossings: {early_crossings}')
print(f'good crossings: {good_crossings}')
print(f'total crossings: {crossing_trials} (good + early + wrong + missing)')

# decoded onset latency from crossing_bin: T16 relative to go_bin, T11 relative to trial start
if participant == 'T16':
    rel_crossing_times = np.where(dec_onset_bins != -1,
                                  (dec_onset_bins - go_bins) * TS, np.nan)
else:
    rel_crossing_times = np.where(dec_onset_bins != -1,
                                  (dec_onset_bins - start_bin) * TS, np.nan)

# filter out NaN values (no decoded onset)
rel_crossing_times = rel_crossing_times[~np.isnan(rel_crossing_times)]

# histogram plot params
if participant == 'T16':
    lims_hist = [-800, 1200]
    lims_plot = [-200, 1200]
    xticks = [-200, 0, 400, 800, 1200]
elif participant == 'T11':
    lims_hist = [0, 1200]
    lims_plot = [0, 1200]
    xticks = [-200, 0, 400, 800, 1200]

# compute histogram and cumulative distribution
bin_histogram = 20
t_hist = np.arange(lims_hist[0], lims_hist[1]+bin_histogram, bin_histogram)
counts, _ = np.histogram(rel_crossing_times, t_hist)
cumsum_counts = np.cumsum(counts)/crossing_trials

# generate plot
plt.figure(figsize=(6, 3.2), facecolor='w')

# plot histogram and cumulative distribution
plt.plot(t_hist[:-1]+TS/2, 100*0.95*counts/counts.max(), color=(40/255,170/255,225/255), alpha=0.75, linewidth=2)
plt.plot(t_hist[:-1]+TS/2, 100*cumsum_counts, color='k', alpha=1, linewidth=4)
plt.axvline(x=0, color='g', linestyle='-', linewidth=6)

# misc. plot settings
plt.xlim(lims_plot[0],lims_plot[1])
plt.ylim(0, 101)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.xticks(xticks, fontsize=14)
plt.yticks(np.arange(0, 100 + 1, 20), fontsize=14)
if participant == 'T16':
    plt.xlabel('Time (ms) relative to go cue', fontsize=16)
elif participant == 'T11':
    plt.xlabel('Time (ms) relative to target cue', fontsize=16)
plt.ylabel('Movement onset\ndetection %', fontsize=16)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig5_{participant}_move_onset.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Plot decoded target directions and angular errors

# load trialized data dict
with h5py.File(f'./data/fig5_{participant}_start.h5', 'r') as f:
    eval_trials = {k: f[k][()] for k in f}

# get true and decoded target positions
true_targets = eval_trials['target_true_all']
pred_targets = eval_trials['target_pred_all']

# compute angles (0 - 360 degrees)
true_angles = np.arctan2(true_targets[:, 1], true_targets[:, 0])
true_angles = np.where(true_angles < 0, true_angles + 2*np.pi, true_angles)
pred_angles = np.arctan2(pred_targets[:, 1], pred_targets[:, 0])
pred_angles = np.where(pred_angles < 0, pred_angles + 2*np.pi, pred_angles)

# compute angular errors (0 - 180 degrees)
angle_diff = np.abs(pred_angles - true_angles)
angle_errors = np.where(angle_diff > np.pi, 2*np.pi - angle_diff, angle_diff)

# filter to only good crossings
target_true_all = [true_targets[tr] for tr in good_crossings]
target_pred_all = [pred_targets[tr] for tr in good_crossings]
angle_true_all  = [true_angles[tr] for tr in good_crossings]
angle_pred_all  = [pred_angles[tr] for tr in good_crossings]
angle_error_all = [angle_errors[tr] for tr in good_crossings]

# plot decoded target directions with jitter and success/failure markers
plt.figure(figsize=(6, 6))

# draw circles for each target direction
radius = 120
for i in range(8):
    angle = i * (2 * np.pi / 8)
    c = colors.hsv_to_rgb([angle / (2 * np.pi), 1, 1])
    circle = plt.Circle((400*np.cos(angle), 400*np.sin(angle)),
                        radius, color=c, fill=False, linestyle='-', linewidth=2)
    plt.gca().add_artist(circle)

# counters for successful and failed trials
n_succesful = 0
n_failed = 0

np.random.seed(42)
# plot all trials with success/fail markers and distance jitter
for tr in range(len(target_true_all)):

    angle_pred = angle_pred_all[tr]
    target_pred = target_pred_all[tr]
    angle_true = angle_true_all[tr]
    target_true = target_true_all[tr]

    # add distance jitter to the predicted target position for visualization
    jitter = np.random.uniform(-15, 15)
    target_pred_jitter = np.array([np.cos(angle_pred),
                            np.sin(angle_pred)]) * (400 + jitter)

    # determine if trial is successful (within radius) or failed (outside radius)
    if np.linalg.norm(target_pred - target_true) > radius:
        marker = 'x'
        s = 30
        linewidth = 2
        n_failed += 1
    else:
        marker = 'o'
        s = 35
        linewidth = 0
        n_succesful += 1

    # draw trial marker
    c = colors.hsv_to_rgb([angle_true / (2 * np.pi), 1, 1])
    plt.scatter(target_pred_jitter[0], target_pred_jitter[1],
            c=c, marker=marker, s=s, alpha=0.8, linewidth=linewidth)

print(f'n success: {n_succesful}')
print(f'n failed: {n_failed}')
print(f'success rate: {(n_succesful)/(n_succesful+n_failed)}')

# misc. plot settings
plt.xlim(-550, 550)
plt.ylim(-550, 550)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().spines[['top', 'right', 'left', 'bottom']].set_visible(False)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig5_{participant}_decoded_directions.pdf')
plt.savefig(savepath, format='pdf')

plt.show()

# compute median and mean angular error in degrees
median_error = np.median(angle_error_all) * 180 / np.pi
mean_error = np.mean(angle_error_all) * 180 / np.pi
print(f'{len(angle_error_all)} trials')
print(f'Median angular error: {median_error:.2f} degrees')
print(f'Mean angular error: {mean_error:.2f} degrees')

# plot histogram of angular errors
plt.figure(figsize=(5, 2))
plt.hist(np.array(angle_error_all)*180/np.pi, bins=np.arange(0, 180, 5))
plt.axvline(median_error, color='k', linestyle='--', label='median', linewidth=2)

# misc. plot settings
plt.xlim(0, 180)
plt.ylim(0, 25)
plt.xticks(np.arange(0, 180 +1, 20), fontsize=11)
plt.yticks(np.arange(0, 25 + 1, 5), fontsize=11)
plt.xlabel('Angular error (degrees)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.gca().spines[['top', 'right']].set_visible(False)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig5_{participant}_angular_errors.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Load prep decoding and OLE trajectories for comparison

# cursor trajectories from the optimal-linear-estimator sessions, the baseline the
# preparatory paradigm is compared against below (near targets < 100 px excluded)

# load trialized data dict for prep decoding
with h5py.File(f'./data/fig5_{participant}_start.h5', 'r') as f:
    prep_trials = {k: f[k][()] for k in f}

# generate list of cursor trayectories (movement start cue to end) for each prep decoding trial
cursor_pos_all = []
for tr in good_crossings:
    trial_start = go_bins[tr] if participant == 'T16' else start_bin
    trial_end = trial_end_bins[tr]
    cursor_pos_tr = prep_trials['cursor_data_all'][tr, trial_start:trial_end, :]
    # pad the end of the trajectory with the last position
    cursor_pos_tr = np.concatenate((cursor_pos_tr, np.tile(cursor_pos_tr[-1, :], (100, 1))), axis=0)
    cursor_pos_all.append(cursor_pos_tr)

# load trialized data dict for OLE decoding
with h5py.File(f'./data/fig5_{participant}_start_ole.h5', 'r') as f:
    ole_trials = {k: f[k][()] for k in f}

# get cursor positions, go cue bins, end bins, and target positions for OLE trials
ole_cursor = ole_trials['cursor_data_all']
ole_go_bins = ole_trials['go_bin_all'].astype(int)
ole_end_bins = ole_trials['end_bin_all'].astype(int)
ole_targets = ole_trials['target_true_all']

# generate list of cursor trayectories (movement start cue to end) and target positions for each OLE decoding trial
cursor_pos_all_2 = []
target_pos_all_2 = []
for tr in range(len(ole_go_bins)):
    cursor_pos = ole_cursor[tr, ole_go_bins[tr]:ole_end_bins[tr], :]
    target_pos = ole_targets[tr]
    # only use trials with targets > 100 pixels away from the center
    if np.linalg.norm(target_pos - np.array([0, 0])) > 100:
        cursor_pos_all_2.append(cursor_pos)
        target_pos_all_2.append(target_pos)


# %%
## Compute movement trajectory metrics

def compute_target_acquired(distance_to_target, radius, hold_bins):
    """
    Return index of when the target was acquired (hold_bin inside radius)
    """
    i_first_on_target = 9999
    on_target = False
    for i in range(len(distance_to_target)):
        # on target
        if distance_to_target[i] <= radius:
            # change state to in target
            if not on_target:
                on_target = True
                i_first_on_target = i
            # check if hold time is reached
            elif i - i_first_on_target >= hold_bins:
                # return the index of the target acquisition
                return i
        # not on target
        else:
            on_target = False
    # return NaN if the target was never acquired
    return np.nan

def compute_time_to_target(cursor_pos, target_pos, radius=60):
    """
    Compute the time when the cursor first reaches the target.
    """
    distances = np.linalg.norm(cursor_pos - target_pos, axis=1)
    time_to_target = np.where(distances <= radius)[0]
    if len(time_to_target) > 0:
        # return the first time the cursor is within the target
        return time_to_target[0]
    else:
        # if the target was never reached
        return np.nan

# target radius for success (normalized to move_dist)
SUCCESS_RADIUS = 0.3
# bin width in seconds, and the 500 ms hold to acquire the target
TS = BIN_WIDTH
HOLD_BINS = int(0.5/TS)

# dict of parameters for each decoder type
decoder_dict = {
    'prep': dict(cursor=cursor_pos_all,
                 targets=target_true_all,
                 move_dist=400,
                 color=(0.25, 0.25, 1),
                 label='Preparatory \ncontrol paradigm'),
    'ole':  dict(cursor=cursor_pos_all_2,
                 targets=target_pos_all_2,
                 # one of the OLE sessions had a slightly longer move distance, so normalzie accordingly
                 move_dist=460 if participant == 'T16' else 400,
                 color=(0.85, 0.25, 0.25),
                 label='Optimal linear\nestimator'),
}

# loop over decoder types, collecting per-trial metrics
metrics = {}
for name, cond in decoder_dict.items():

    # containers for per-trial metrics
    time_to_target = []
    distance_to_target = []
    target_acquired_times = []
    successful_trials = []
    failed_trials = []

    # loop over trials
    for tr, cursor_pos in enumerate(cond['cursor']):
        target_pos = cond['targets'][tr]

        dist = np.linalg.norm(cursor_pos - target_pos, axis=1)

        # if trial was successful (cursor end within SUCCESS_RADIUS of target), compute metrics
        if dist[-1]/cond['move_dist'] < SUCCESS_RADIUS:
            successful_trials.append(tr)
            # compute time to target
            time_to_target.append(TS*compute_time_to_target(cursor_pos/cond['move_dist'],
                                                            target_pos/cond['move_dist'],
                                                            radius=SUCCESS_RADIUS))
            # compute distance to target through time
            distance_to_target.append(np.linalg.norm(cursor_pos/cond['move_dist'] - target_pos/cond['move_dist'], axis=1))
            # compute target acquisition time
            target_acquired_times.append(compute_target_acquired(distance_to_target[-1], SUCCESS_RADIUS, HOLD_BINS))
        else:
            failed_trials.append(tr)

    print(f'{name}: {len(successful_trials)} successful, {len(failed_trials)} failed trials')

    # stack the per-trial distance traces, NaN-padded to the longest trial
    max_time = np.max([len(tr) for tr in distance_to_target])
    dist_to_target_stack = np.full((len(distance_to_target), max_time), np.nan)
    for i, tr in enumerate(distance_to_target):
        dist_to_target_stack[i, :len(tr)] = tr[:]

    # write metrics to dict for plotting
    metrics[name] = dict(time_to_target=time_to_target,
                         target_acquired_times=target_acquired_times,
                         dist_to_target_stack=dist_to_target_stack,
                         successful_trials=successful_trials,
                         failed_trials=failed_trials,
                         color=cond['color'], label=cond['label'])


# %%
## Plot trajectory metrics — preparatory paradigm vs. OLE

# draw OLE first so the prep traces sit on top of it (also sets the legend order)
plot_order = ('ole', 'prep')

# 1. plot histogram of time to target (successful trials only)
plt.figure(figsize=(5, 2), facecolor='w')

# define histogram bins for time to target
bin_histogram = 0.1
t_hist = np.arange(0, 10+bin_histogram, bin_histogram)

# time to target histograms
for name in plot_order:
    m = metrics[name]
    plt.hist(m['time_to_target'], bins=t_hist, color=m['color'], alpha=0.5, density=True)
    plt.axvline(np.mean(m['time_to_target']), color=m['color'], linestyle='-', label='median', linewidth=3)

for name in plot_order:
    print(f'{name} mean time to target: {np.mean(metrics[name]["time_to_target"])}')

# misc. plot settings
plt.xlabel('Time to target (s)', fontsize=14)
plt.xlim([0,3])
plt.ylim([0, 4])
plt.gca().spines[['top', 'right', 'left']].set_visible(False)
plt.gca().yaxis.set_ticks([])
plt.gca().tick_params(axis='both', which='major', labelsize=11)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig5_{participant}_time_to_target.pdf')
plt.savefig(savepath, format='pdf')

plt.show()

# 2. plot distance to target through time (successful trials only)
plt.figure(figsize=(6.4, 4.8))

# Mann-Whitney U test (Wilcoxon rank-sum) for time to target between decoders
stat, p_value = mannwhitneyu(metrics['prep']['time_to_target'],
                             metrics['ole']['time_to_target'], alternative='less')
print("Mann-Whitney U statistic:", stat)
print("p-value:", p_value)

# plot distance to target traces
for name in plot_order:
    m = metrics[name]
    dist_to_target_tr = m['dist_to_target_stack']
    acquired_time_tr = m['target_acquired_times']
    # loop over trials, plotting each trial's distance to target trace up to the acquisition time
    for i in range(dist_to_target_tr.shape[0]):
        end_i = acquired_time_tr[i]
        plt.plot(np.arange(0, dist_to_target_tr[i,:end_i].shape[0])*TS,
                dist_to_target_tr[i,:end_i], color=m['color'], alpha=0.1, linewidth=2)
        plt.scatter((end_i-1)*TS, dist_to_target_tr[i,end_i-1], color=m['color'], alpha=0.6, s=20, linewidth=0)

# plot mean distance to target traces
for name in plot_order:
    m = metrics[name]
    dist_to_target_tr = m['dist_to_target_stack']
    acquired_time_tr = m['target_acquired_times']
    end_i = int(np.floor(np.nanmean(acquired_time_tr)))
    plt.plot(np.arange(0, end_i)*TS,
             np.nanmean(dist_to_target_tr[:,:end_i], axis=0), color=m['color'], alpha=1, linewidth=4, label=m['label'])

# draw line at success radius
plt.axhline(y=SUCCESS_RADIUS, color='k', linestyle='--', linewidth=1)
plt.text(0.25, 0.29, 'on target', va='top', ha='center', fontsize=11)

# misc. plot settings
plt.gca().tick_params(axis='both', which='major', labelsize=11)
plt.xlim([0,3])
plt.ylim(0, 1.5)
plt.xlabel('Time after instructed movement start (s)', fontsize=14)
plt.ylabel('Distance to target\n(% of total distance)', fontsize=14)
plt.legend(fontsize=11)
plt.gca().spines[['top', 'right']].set_visible(False)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig5_{participant}_dist_to_target.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Run channel count sweep — target-direction decoding

# whether to run the channel sweep or just use cached .pkl results to plot
RUN_CHAN_SWEEP  = True

# random channel subsets per count
N_REPS = 20

# reduce the flattened neural features via PCA before the ridge decode
# (T16 only, mirroring the deployed decoders)
USE_SWEEP_PCA = (participant == 'T16')
SWEEP_PCA_VAF = 0.90

def pca_reduce(neural_train, neural_test, vaf=SWEEP_PCA_VAF):
    """
    Project train and test onto the PCA components (fit on train) explaining
    `vaf` cumulative variance.
    """
    pca = PCA(n_components=vaf, svd_solver='full').fit(neural_train)
    return pca.transform(neural_train), pca.transform(neural_test)

def circular_angular_error(true_ang, pred_angle):
    """
    Compute the circular angular error between true and predicted angles.
    Returns mean and median angular error in degrees.
    """
    diff = np.abs(true_ang - pred_angle)
    diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
    return np.mean(diff) * 180 / np.pi, np.median(diff) * 180 / np.pi

if not RUN_CHAN_SWEEP:
    print('RUN_CHAN_SWEEP is False — skipping channel sweep, using cached .pkl.')
else:

    # load data needed for the channel sweep (neural features, target angles, and eval block bools)
    with h5py.File(f'./data/fig5_{participant}_onset.h5', 'r') as f:
        sweep_data = {k: f[k][()] for k in f}

    # get the t=0 bin index
    center_bin = int(np.where(sweep_data['t_data'] == 0)[0][0])

    # get decoder params
    keep_channels = sweep_data['keep_chans_all']
    prep_bins = int(prep_window_len / (1000 * BIN_WIDTH))

    def get_prep_features(is_eval_val):
        """
        Return prep-period neural features and target angles for the specified trial mask.
        """
        mask = sweep_data['eval_block_bool_all'] == is_eval_val
        # get prep-period neural features (shape: n_trials x prep_bins x n_channels)
        neural_features = sweep_data['neural_data_norm_all'][mask][:, center_bin - prep_bins + 1:center_bin + 1, :].astype(float)
        # get target angles
        target_true = sweep_data['target_true_all'][mask]
        angle = np.arctan2(target_true[:, 1], target_true[:, 0]) % (2 * np.pi)
        # return neural features, target unit vectors, and angles
        return neural_features, np.stack([np.cos(angle), np.sin(angle)], axis=1), angle

    # get train and test features
    neural_train, cossin_train, _            = get_prep_features(0)
    neural_test,  _,              angle_test = get_prep_features(1)
    print(f'Training trials: {neural_train.shape[0]}   Test trials: {neural_test.shape[0]}')

    # total number of channels (spikes + SBP) and electrodes
    n_channels = neural_train.shape[2]
    n_electrodes  = n_channels // 2

    # select electrodes and counts for sweep
    keep_electrodes  = keep_channels[keep_channels < n_electrodes]
    n_keep_electrodes        = len(keep_electrodes)
    CHANNEL_COUNTS = np.unique(np.round(
        np.logspace(np.log10(4), np.log10(n_keep_electrodes), 15)
    ).astype(int))

    # containers for angular error results (mean and median) for each channel count and repetition
    ang_err_mean_sweep = np.full((len(CHANNEL_COUNTS), N_REPS), np.nan)
    ang_err_med_sweep  = np.full((len(CHANNEL_COUNTS), N_REPS), np.nan)

    # loop over channel counts and repetitions, training and testing a Ridge decoder on random channel subsets
    rng = np.random.default_rng(seed=42)
    ridge_alphas = np.logspace(-1, 5, 25)
    for count_idx, n_chan in enumerate(CHANNEL_COUNTS):
        for rep in range(N_REPS):

            # randomly select n_chan electrodes (and their corresponding SBP channels)
            electrode_subset = keep_electrodes[rng.choice(n_keep_electrodes, n_chan, replace=False)]
            channel_subset  = np.concatenate([electrode_subset, electrode_subset + n_electrodes])
            # flatten time and channel dimensions for Ridge regression
            neural_train_sub = neural_train[:, :, channel_subset].reshape(neural_train.shape[0], -1)
            neural_test_sub = neural_test[:, :, channel_subset].reshape(neural_test.shape[0], -1)
            # optionally reduce dimensionality via PCA before Ridge regression
            if USE_SWEEP_PCA:
                neural_train_sub, neural_test_sub = pca_reduce(neural_train_sub, neural_test_sub)
            # fit decoder (neural -> target unit vector) with internal cross-validation to select the best regularization parameter (alpha)
            decoder = Pipeline([('ridge', RidgeCV(alphas=ridge_alphas))])
            try:
                decoder.fit(neural_train_sub, cossin_train)
            except Exception:
                print(f'Failed to fit decoder for n_chan={n_chan}, rep={rep}. Skipping this repetition.')
                continue
            # compute predicted target unit vectors and angles for the test set, then compute angular error metrics
            pred_cossin = decoder.predict(neural_test_sub)
            pred_angle  = np.arctan2(pred_cossin[:, 1], pred_cossin[:, 0]) % (2 * np.pi)
            ang_err_mean_sweep[count_idx, rep], ang_err_med_sweep[count_idx, rep] = circular_angular_error(angle_test, pred_angle)
        print(f'n_chan={n_chan:4d}  '
              f'ang_err(mean)={np.nanmean(ang_err_mean_sweep[count_idx]):.1f}°  '
              f'ang_err(med)={np.nanmean(ang_err_med_sweep[count_idx]):.1f}°')

    # save results
    pkl_path = os.path.join(save_plot_filepath, f'fig5_{participant}_channel_sweep.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'CHANNEL_COUNTS':      CHANNEL_COUNTS,
            'ang_err_mean_sweep':  ang_err_mean_sweep,
            'ang_err_med_sweep':   ang_err_med_sweep,
            'N_REPS':              N_REPS,
            'participant':         participant,
        }, f)


# %%
## Plot channel count sweep results

# metric to plot
ANG_ERR_METRIC = 'median'  # 'mean' or 'median'

# extrapolation target channel count for the power-law fit
PROJ_CHANS   = 1000

# check for pkl results from the channel sweep, and load them if they exist.  Otherwise, raise an error.
if not os.path.exists(os.path.join(save_plot_filepath, f'fig5_{participant}_channel_sweep.pkl')):
    raise FileNotFoundError(f'Channel sweep results not found for {participant}. '
                            f'Run the channel sweep first (RUN_CHAN_SWEEP=True).')
else:

    # load results
    print(f'Loading channel sweep results for {participant} from .pkl file.')
    pkl_path = os.path.join(save_plot_filepath, f'fig5_{participant}_channel_sweep.pkl')
    with open(pkl_path, 'rb') as f:
        sweep_results = pickle.load(f)

    CHANNEL_COUNTS     = sweep_results['CHANNEL_COUNTS']
    N_REPS             = sweep_results['N_REPS']
    ang_sweep = sweep_results['ang_err_med_sweep'] if ANG_ERR_METRIC == 'median' else sweep_results['ang_err_mean_sweep']

    # compute mean and standard error of the mean (SEM) across repetitions for each channel count
    ang_mean = np.nanmean(ang_sweep, axis=1)
    ang_sem  = np.nanstd(ang_sweep,  axis=1) / np.sqrt(N_REPS)


    # fit a power-law (y = a * x^b) to the mean angular error vs. channel count (linear in log-log space)
    x_counts = CHANNEL_COUNTS.astype(float)
    x_fit = np.logspace(np.log10(x_counts[0]), np.log10(PROJ_CHANS), 200)
    pw_exp, pw_loga = np.polyfit(np.log(x_counts), np.log(ang_mean), 1)
    y_fit = np.exp(pw_loga) * x_fit ** pw_exp
    print(f'power-law fit: exponent={pw_exp:.2f}  '
        f'@ {PROJ_CHANS} ch → {np.exp(pw_loga)*PROJ_CHANS**pw_exp:.1f}°')

    # generate plot
    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor='w')

    # plot mean ± SEM angular error vs. channel count
    ax.fill_between(CHANNEL_COUNTS,
                    ang_mean - ang_sem, ang_mean + ang_sem,
                    alpha=0.3, color=(0.25, 0.25, 1))
    ax.plot(CHANNEL_COUNTS, ang_mean, 'o-',
            color=(0.25, 0.25, 1), linewidth=2, markersize=5)

    # plot result for the full channel count (last point) with a larger marker
    ax.plot(CHANNEL_COUNTS[-1], ang_mean[-1], 'o',
            color=(0.25, 0.25, 1), markersize=12, markeredgewidth=0, zorder=5)

    # plot power-law fit
    ax.plot(x_fit, y_fit, ':', color='gray', linewidth=1.5)

    # misc. plot settings
    ax.set_xscale('log')
    ax.set_xlabel('Number of active channels', fontsize=14)
    ax.set_ylabel('Median angular error (°)', fontsize=14)
    ax.set_ylim(0, 90)
    ax.set_xlim(left=CHANNEL_COUNTS[0] * 0.8, right=PROJ_CHANS * 1.3)
    # regular decade ticks within the visible range
    x_ticks = [t for t in [10, 100, 1000] if CHANNEL_COUNTS[0]*0.8 <= t <= PROJ_CHANS*1.3]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks])
    ax.tick_params(labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)

    # save pdf
    plt.tight_layout()
    savepath = os.path.join(save_plot_filepath,
                            f'fig5_{participant}_channel_sweep.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
## 2D window sweep — decoder performance vs. window start and end time relative to decoder
## movement onset (only upper-triangle cells, start < end, are valid)

# only run for T16
RUN_WINDOW_SWEEP    = (participant == 'T16')

# reduce the flattened neural features via PCA before the ridge decode
# (T16 only, mirroring the deployed decoders)
USE_SWEEP_PCA = (participant == 'T16')
SWEEP_PCA_VAF = 0.90

if not RUN_WINDOW_SWEEP:
    print('Skipping window sweep (RUN_WINDOW_SWEEP is False).')
else:

    # load data needed for the window sweep (neural features, target angles, and eval block bools)
    with h5py.File(f'./data/fig5_{participant}_onset.h5', 'r') as f:
        sweep_data = {k: f[k][()] for k in f}

    SWEEP_MS   = np.arange(-800, 801, 100)
    n_sweep    = len(SWEEP_MS)

    # get the t=0 (decoded onset) bin index
    center_bin = int(np.where(sweep_data['t_data'] == 0)[0][0])

    # select same channels use in the real decoder (spike + SBP pair per electrode)
    keep_channels = sweep_data['keep_chans_all']
    n_channels = sweep_data['neural_data_norm_all'].shape[2]
    n_electrodes  = n_channels // 2
    keep_electrodes  = keep_channels[keep_channels < n_electrodes]
    kept_channel_subset = np.concatenate([keep_electrodes, keep_electrodes + n_electrodes])

    def get_window_features(is_eval_val, t_start_ms, t_end_ms, BIN_WIDTH):
        """
        Get flattened neural features buffer [crossing+start : crossing+end] per trial.
        """
        mask = sweep_data['eval_block_bool_all'] == is_eval_val
        start_bins = int(round(t_start_ms / (1000 * BIN_WIDTH)))
        end_bins   = int(round(t_end_ms   / (1000 * BIN_WIDTH)))
        # for diagonal cells (start == end) add 1 bin to the end to avoid empty slices
        if end_bins == start_bins:
            end_bins = start_bins + 1
        # get neural features for the specified window and channel subset
        neural_slice = sweep_data['neural_data_norm_all'][mask][:, center_bin + start_bins:center_bin + end_bins, kept_channel_subset]
        neural_features = neural_slice.reshape(neural_slice.shape[0], -1).astype(float)
        # get target angle
        targets = sweep_data['target_true_all'][mask]
        angle = np.arctan2(targets[:, 1], targets[:, 0]) % (2 * np.pi)
        # only keep trials with no NaNs in the neural features
        in_bounds = ~np.isnan(neural_slice).any(axis=(1, 2))
        neural_features, angle = neural_features[in_bounds], angle[in_bounds]
        # return flattened neural features, target unit vectors, and angles
        return neural_features, np.stack([np.cos(angle), np.sin(angle)], axis=1), angle

    # containers for angular error results (mean and median) for each window start/end combination
    ang_err_mean_2d = np.full((n_sweep, n_sweep), np.nan)
    ang_err_med_2d  = np.full((n_sweep, n_sweep), np.nan)

    # loop over all window start/end combinations, training and testing a Ridge decoder for each
    ridge_alphas = np.logspace(-1, 5, 25)
    for si, t_start_ms in enumerate(SWEEP_MS):
        for ei, t_end_ms in enumerate(SWEEP_MS):
            # lower triangle — invalid
            if t_end_ms < t_start_ms:
                continue
            # get train and test features
            neural_train, cossin_train, _     = get_window_features(0, t_start_ms, t_end_ms, BIN_WIDTH)
            neural_test, _,        angle_test = get_window_features(1, t_start_ms, t_end_ms, BIN_WIDTH)
            # skip if too few trials for train or test
            if neural_train.shape[0] < 10 or neural_test.shape[0] < 5:
                continue
            # optionally reduce dimensionality via PCA before Ridge regression
            if USE_SWEEP_PCA:
                neural_train, neural_test = pca_reduce(neural_train, neural_test)
            decoder = Pipeline([('ridge', RidgeCV(alphas=ridge_alphas))])
            # fit decoder (neural -> target unit vector) with internal cross-validation to select the best regularization parameter (alpha)
            try:
                decoder.fit(neural_train, cossin_train)
            except Exception:
                print(f'Failed to fit decoder for t_start={t_start_ms}, t_end={t_end_ms}. Skipping this window.')
                continue
            # compute predicted target unit vectors and angles for the test set, then compute angular error metrics
            pred_cossin = decoder.predict(neural_test)
            pred_angle  = np.arctan2(pred_cossin[:, 1], pred_cossin[:, 0]) % (2 * np.pi)
            ang_err_mean_2d[si, ei], ang_err_med_2d[si, ei] = \
                circular_angular_error(angle_test, pred_angle)
        err_row = ang_err_med_2d[si] if ANG_ERR_METRIC == 'median' else ang_err_mean_2d[si]
        if np.any(~np.isnan(err_row)):
            print(f't_start={t_start_ms:+5d} ms  '
                  f'best_ang_err={np.nanmin(err_row):.1f}°')

    # save results
    pkl_path = os.path.join(save_plot_filepath, f'fig5_{participant}_window_sweep.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'SWEEP_MS':           SWEEP_MS,
            'ang_err_mean_2d':    ang_err_mean_2d,
            'ang_err_med_2d':     ang_err_med_2d,
            'participant':        participant,
        }, f)


# %%
## Plot window sweep results

# metric to plot
ANG_ERR_METRIC     = 'median'  # 'mean' or 'median'

# check for pkl results from the window sweep, and load them if they exist.
if os.path.exists(os.path.join(save_plot_filepath, f'fig5_{participant}_window_sweep.pkl')):

    # load results
    print(f'Loading window sweep results for {participant} from .pkl file.')
    pkl_path = os.path.join(save_plot_filepath, f'fig5_{participant}_window_sweep.pkl')
    with open(pkl_path, 'rb') as f:
        sweep_results = pickle.load(f)

    SWEEP_MS           = sweep_results['SWEEP_MS']
    ang_err_2d         = sweep_results['ang_err_med_2d'] if ANG_ERR_METRIC == 'median' else sweep_results['ang_err_mean_2d']

    # restrict to prep window (start: -800→0) × latency window (end: 0→+800)
    onset_idx = np.where(SWEEP_MS == 0)[0][0]
    row_ms  = SWEEP_MS[:onset_idx + 1]
    col_ms  = SWEEP_MS[onset_idx:]
    ang_plot = ang_err_2d[:onset_idx + 1, onset_idx:]

    # ticks every 400 ms on both axes
    tick_step_ms  = 400
    ms_per_bin    = int(round(SWEEP_MS[1] - SWEEP_MS[0]))
    tick_every = tick_step_ms // ms_per_bin
    row_tick_idx   = list(range(0, len(row_ms), tick_every))
    col_tick_idx   = list(range(0, len(col_ms), tick_every))
    row_tick_labels   = [f'{ms:+d}' for ms in row_ms[row_tick_idx]]
    col_tick_labels   = [f'{ms:+d}' for ms in col_ms[col_tick_idx]]

    # generate plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor='w')

    # plot the angular error heatmap
    vmin = 8.5
    vmax = 20
    im = ax.imshow(ang_plot.T, origin='lower', aspect='equal',
                   cmap='viridis_r', vmin=vmin, vmax=vmax,
                   interpolation='nearest')

    # add colorbar to the right of the heatmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.25)
    cb = plt.colorbar(im, cax=cax)
    cb.set_ticks([10, 15, 20])
    cb.set_label('Angular error (°)', fontsize=14)
    cb.ax.tick_params(labelsize=12)

    # add text annotations to the heatmap
    for r in range(ang_plot.T.shape[0]):
        for c in range(ang_plot.T.shape[1]):
            val = ang_plot.T[r, c]
            if not np.isnan(val):
                norm_val = (val - vmin) / (vmax - vmin)  # viridis_r: high val → dark → white text
                tc = 'w' if norm_val > 0.5 else 'k'
                ax.text(c, r, f'{val:.1f}', ha='center', va='center', fontsize=10, color=tc)

    # misc. plot settings
    ax.set_xticks(row_tick_idx)
    ax.set_xticklabels(row_tick_labels, rotation=0, fontsize=12)
    ax.set_yticks(col_tick_idx)
    ax.set_yticklabels(col_tick_labels, fontsize=12)
    ax.tick_params(axis='y', labelrotation=90, labelsize=12)
    for lbl in ax.get_yticklabels():
        lbl.set_va('center')
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel('Decoding window start\n(ms rel. to movement onset)', fontsize=14)
    ax.set_ylabel('Decoding window end\n(ms rel. to movement onset)', fontsize=14)
    ax.set_title(f'{participant}\nMedian angular error for\ndifferent decoding windows', fontsize=16, pad=15)

    plt.tight_layout()

    # save pdf
    savepath = os.path.join(save_plot_filepath,
                            f'fig5_{participant}_window_sweep.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()

