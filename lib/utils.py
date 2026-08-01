"""Utility functions used by the figure scripts (fig1-fig5.py): 
HDF5 loader, neural data normalization, neural data marginalization per condition
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import itertools


def load_trial_data_h5(filepath):
    """Load a trialized data dict from HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.

    Returns
    -------
    trial_data : dict
        A dict mapping each key to its numpy data array.
    """

    trial_data = {}

    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            
            # load data for each key
            v = f[key][()]
            
            # decode bytes to string if necessary
            if isinstance(v, np.ndarray) and v.dtype.kind == 'S':
                v = np.char.decode(v, 'utf-8')
            
            # store data in the trial_data dict
            trial_data[key] = v

    # return trialized data dict
    return trial_data


def normalize_data(neural_data_delay_all, block_id_data_all, neural_data_move_all=None,
                   SOFT_NORM=2, center_baseline=True, use_move_data=True,
                   GO_CUE=50, TS=20, plot=False,
                   ):
    """Normalize neural data per block.

    Parameters
    ----------
    neural_data_delay_all : ndarray, shape (trials, T, N_ch)
        Neural features (not normalized) for the delay epoch.
    block_id_data_all : ndarray, shape (trials,)    
        Array indicating the block ID for each trial.
    neural_data_move_all : ndarray, shape (trials, T, N_ch)
        Neural features (not normalized) for the movement epoch.
    SOFT_NORM : float
        Soft-normalization constant (added to std).
    center_baseline : bool
        Whether to use the baseline period (pre-cue) to center the normalized data.
        If False, the mean across the whole window is used for centering instead.
    use_move_data : bool
        Whether to combine the delay and movement data to compute the std for normalization. 
        If False, the stds for scaling delay and movement data are computed separately.
    GO_CUE : int
        Index of the alignment cue for each epoch in the data arrays (assumed to be the same).
    TS : float
        Per-bin time step in ms.
    plot : bool
        Whether to plot the means and stds across blocks (diagnostic plots).

    Returns
    -------
    neural_data_delay_norm_all, neural_data_move_norm_all : ndarray, shape (trials, T, N_ch)
        Normalized neural data arrays.
    """

    # data range to use for normalization window
    T_STA = GO_CUE+int(-200/TS)
    T_END = GO_CUE+int(1000/TS)

    # containers for normalized data
    neural_data_delay_norm_all = np.zeros_like(neural_data_delay_all)
    neural_data_move_norm_all = np.zeros_like(neural_data_move_all)

    # per-block mean/baseline subtraction
    for block in np.unique(block_id_data_all):
        block_mask = block_id_data_all == block

        if center_baseline:
            # use baseline period (T_STA -> 0) to compute mean for centering
            neural_mean = neural_data_delay_all[block_mask][:,T_STA:GO_CUE,:].mean(axis=(0,1))
        else:
            # use whole windows (T_STA -> T_END) of delay and (0 -> 1000 ms) of movement to compute mean for centering
            neural_mean = np.concatenate([neural_data_delay_all[block_mask][:,T_STA:T_END,:],
                                neural_data_move_all[block_mask][:,GO_CUE:GO_CUE+int(1000/TS),:]], axis=1).mean(axis=(0,1))
        
        # compute and save mean-subtracted delay/move data for corresponding block
        idx = np.ix_(block_mask,range(neural_data_delay_all.shape[1]),range(neural_data_delay_all.shape[2]))
        neural_data_delay_norm_all[idx] = (neural_data_delay_all[idx] - neural_mean)
        idx_move = np.ix_(block_mask,range(neural_data_move_all.shape[1]),range(neural_data_move_all.shape[2]))
        neural_data_move_norm_all[idx_move] = (neural_data_move_all[idx_move] - neural_mean)

    # use the joint std of delay and movement data for normalization
    if use_move_data:
        neural_std = np.concatenate([neural_data_delay_norm_all[:,GO_CUE:T_END,:],
                                    neural_data_move_norm_all[:,GO_CUE:T_END,:]], axis=1).std(axis=(0,1))
        neural_data_delay_norm_all /= (neural_std + SOFT_NORM)
        neural_data_move_norm_all /= (neural_std + SOFT_NORM)
    # separately compute stds to normalize delay and movement data
    else:
        neural_std = neural_data_delay_norm_all[:,GO_CUE:T_END,:].std(axis=(0,1))
        neural_data_delay_norm_all /= (neural_std + SOFT_NORM)
        neural_std_move = neural_data_move_norm_all[:,GO_CUE:T_END,:].std(axis=(0,1))
        neural_data_move_norm_all /= (neural_std_move + SOFT_NORM)

    # (diagnostic) compute means/stds for delay period across blocks
    means_array = np.zeros((np.unique(block_id_data_all).shape[0],
                            neural_data_delay_norm_all.shape[2]))
    stds_array = np.zeros((np.unique(block_id_data_all).shape[0],
                            neural_data_delay_norm_all.shape[2]))
    for ib, block in enumerate(np.unique(block_id_data_all)):
        block_mask = block_id_data_all == block
        neural_mean = neural_data_delay_norm_all[block_mask].mean(axis=(0,1))
        neural_std = neural_data_delay_norm_all[block_mask].std(axis=(0,1))
        means_array[ib,:] = neural_mean
        stds_array[ib,:] = neural_std

    # (diagnostic) plot means/stds for delay period across blocks
    if plot:
        plt.plot(means_array)
        plt.show()
        plt.plot(stds_array)
        plt.show()

    # (diagnostic) compute means/stds for movement period across blocks
    means_array = np.zeros((np.unique(block_id_data_all).shape[0],
                            neural_data_move_norm_all.shape[2]))
    stds_array = np.zeros((np.unique(block_id_data_all).shape[0],
                            neural_data_move_norm_all.shape[2]))
    for ib, block in enumerate(np.unique(block_id_data_all)):
        block_mask = block_id_data_all == block
        neural_mean = neural_data_move_norm_all[block_mask].mean(axis=(0,1))
        neural_std = neural_data_move_norm_all[block_mask].std(axis=(0,1))
        means_array[ib,:] = neural_mean
        stds_array[ib,:] = neural_std

    # (diagnostic) plot means/stds for movement period across blocks
    if plot:
        plt.plot(means_array)
        plt.show()
        plt.plot(stds_array)
        plt.show()

    # return normalized data arrays
    return neural_data_delay_norm_all, neural_data_move_norm_all


def normalize_and_concat(spike_data_delay_all, sbp_data_all,
                         spike_data_move_all, sbp_data_move_all,
                         block_id_data_all, GO_CUE, TS,
                         USE_SBP=True,
                         SOFT_NORM_SPIKES=1e-9, SOFT_NORM_SBP=1e-9,
                         plot=False):
    """Normalize time-aligned spike/spike-band power (SBP) data per block and optionally concatenate them.

    Calls ``normalize_data`` independently for spikes and (if ``USE_SBP``) SBP,
    then concatenates along the channel axis.
    
    Parameters
    ----------
    spike_data_delay_all : ndarray, shape (trials, T, N_ch)
        Spike firing rates (not normalized) for the delay epoch.
    sbp_data_all : ndarray (trials, T, N_ch) or None
        SBP data (not normalized) for the delay epoch. Ignored when USE_SBP=False.
    spike_data_move_all : ndarray, shape (trials, T, N_ch)
        Spike firing rates (not normalized) for the movement epoch.
    sbp_data_move_all : ndarray (trials, T, N_ch) or None
        SBP data (not normalized) for the movement epoch. Ignored when USE_SBP=False.
    block_id_data_all : ndarray, shape (trials,)    
        Array indicating the block ID for each trial.
    GO_CUE : int
        Index of the alignment cue for each epoch in the data arrays (assumed to be the same).
    TS : float
        Per-bin time step in ms.
    SOFT_NORM_SPIKES, SOFT_NORM_SBP : float
        Soft-normalization constants for each feature type (added to std).
    plot : bool
        Whether to plot the means and stds across blocks (diagnostic plots).

    Returns
    -------
    neural_data_delay_norm_all, neural_data_move_norm_all : ndarray, shape (trials, T, N_ch[*2 if SBP])
        Normalized (and optionally concatenated) neural data arrays.
        
    """

    # normalize spikes per block (baseline-centered, soft-normalized, separate delay/move normalization)
    spike_delay_norm, spike_move_norm = normalize_data(
        spike_data_delay_all, block_id_data_all,
        neural_data_move_all=spike_data_move_all,
        SOFT_NORM=SOFT_NORM_SPIKES, center_baseline=True, use_move_data=False,
        GO_CUE=GO_CUE, TS=TS, plot=plot
    )

    if USE_SBP:
        # normalize SBP the same way as spikes, then stack onto spikes along channel axis
        sbp_delay_norm, sbp_move_norm = normalize_data(
            sbp_data_all, block_id_data_all,
            neural_data_move_all=sbp_data_move_all,
            SOFT_NORM=SOFT_NORM_SBP, center_baseline=True, use_move_data=False,
            GO_CUE=GO_CUE, TS=TS, plot=plot
        )
        # if USE_SBP=True, return arrays of concatenated normalized spikes+SBP features
        return (np.concatenate([spike_delay_norm, sbp_delay_norm], axis=2),
                np.concatenate([spike_move_norm, sbp_move_norm], axis=2))

    # if USE_SBP=False, only return arrays of normalized spikes features
    return spike_delay_norm, spike_move_norm 


def build_marg_arrays(neural_data_norm_all,
                      CONDITIONS, CONDITION_ARRAYS, trial_mask=None):
    """Build condition-averaged and per-trial arrays, grouped/computed 
    across combinations of each condition.

    Groups trials by every combination of the task condition variables (the
    cartesian product of ``CONDITIONS``), stacks the neural data into a dense
    array indexed by condition, and averages over trials. Supports up to 3
    condition variables. Return format matches the 

    Parameters
    ----------
    neural_data_norm_all : ndarray, shape (trials, T, N_ch)
        Normalized neural data (typically the delay-epoch output of
        ``normalize_and_concat``).
    CONDITIONS : sequence of ndarray
        One array per condition variable, listing that variable's possible values 
        (e.g. the set of target directions). Length 1-3.
    CONDITION_ARRAYS : sequence of ndarray, each shape (trials,)
        Per-trial labels, one array per condition variable, aligned to ``CONDITIONS`` 
        (``CONDITION_ARRAYS[i]`` holds each trial's value for ``CONDITIONS[i]``).
    trial_mask : ndarray of bool, shape (trials,), optional
        Per-trial inclusion mask. Defaults to keeping all trials.

    Returns
    -------
    X : ndarray, shape (N_ch, *cond_dims, T)
        Condition-averaged firing rates (mean over trials) for each condition 
        combination. ``cond_dims`` is one axis per condition variable, sized the 
        number of possible values for that condition.
    trialsX : ndarray, shape (N_ch, *cond_dims, T, max_trials)
        Per-trial data for each condition combination, nan-padded to the max trial 
        count across all conditions.
    trialNum : ndarray, shape (N_ch, *cond_dims)
        Number of valid trials for each condition combination.
    """

    # if no trial mask is provided, default to keeping all trials
    if trial_mask is None:
        trial_mask = np.ones_like(CONDITION_ARRAYS[0]).astype(bool)

    # build a trial mask per condition (cartesian product of condition levels)
    cond_masks = []
    for cond_set in itertools.product(*CONDITIONS):
        masks = [cond_set[i] == CONDITION_ARRAYS[i] for i in range(len(CONDITIONS))]
        mask = np.all(masks, axis=0)
        # drop masked-out trials
        mask = mask & trial_mask
        cond_masks.append(mask)
    
    # compute conditions with min/max # of trials
    min_trials_conds = min([np.sum(cond_masks[i]) for i in range(len(cond_masks))])
    max_trials_conds = max([np.sum(cond_masks[i]) for i in range(len(cond_masks))])
    print(f'min/max trials per conds: {min_trials_conds} - {max_trials_conds}')

    # trialX shape: (max_trials, N_ch, T, *cond_dims)
    trialX_dimensions = (max_trials_conds,
                        neural_data_norm_all.shape[2],
                        neural_data_norm_all.shape[1]) + tuple(len(cond) for cond in CONDITIONS)
    
    # nan-pad: conditions have unequal trial counts
    trialX = np.full(trialX_dimensions, np.nan)   
    
    # trials-per-condition counts
    trialNum = np.zeros(tuple([trialX_dimensions[1]]) + tuple(len(cond) for cond in CONDITIONS))   

    # fill trialX and trialNum one condition at a time
    for cond_set in itertools.product(*CONDITIONS):
        masks = [cond_set[i] == CONDITION_ARRAYS[i] for i in range(len(CONDITIONS))]
        mask = np.all(masks, axis=0)
        mask = mask & trial_mask
        # index of each level along its condition axis
        cond_ix = [np.where(cond == cond_set[i])[0][0] for i, cond in enumerate(CONDITIONS)] 
        idx = np.ix_(*[range(neural_data_norm_all[mask].shape[0]),
                    range(neural_data_norm_all.shape[2]),
                    range(neural_data_norm_all.shape[1])] + [[ix] for ix in cond_ix])
        # transpose (trials, T, N_ch) -> (trials, N_ch, T) and slot into the condition cell of trialX
        if len(CONDITIONS) == 1:
            trialX[idx] = neural_data_norm_all[mask].transpose(0,2,1)[:,:,:,None]
            trialNum[:,cond_ix[0]] = neural_data_norm_all[mask].shape[0]
        elif len(CONDITIONS) == 2:
            trialX[idx] = neural_data_norm_all[mask].transpose(0,2,1)[:,:,:,None,None]
            trialNum[:,cond_ix[0],cond_ix[1]] = neural_data_norm_all[mask].shape[0]
        elif len(CONDITIONS) == 3:
            trialX[idx] = neural_data_norm_all[mask].transpose(0,2,1)[:,:,:,None,None,None]
            trialNum[:,cond_ix[0],cond_ix[1],cond_ix[2]] = neural_data_norm_all[mask].shape[0]
        else:
            raise ValueError('Only up to 3 variables supported')

    # transpose trialsX to (N_ch, *cond_dims, T, trials)
    trialsX = trialX.transpose(tuple([1]) + tuple(cond+3 for cond in range(len(CONDITIONS))) + tuple([2, 0]))
    # trial-average trialsX to get X
    X = np.nanmean(trialsX, axis=-1)   

    # return data arrays grouped by condition combination + trial counts
    return X, trialsX, trialNum