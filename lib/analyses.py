
import itertools
import time
import numpy as np
from joblib import Parallel, delayed

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def compute_crossnobis_matrix(trialsX, trialNum, GO_CUE, T_STA, T_END):
    """Cross-validated Mahalanobis (crossnobis) distance matrix.

    Computes the signed-sqrt leave-one-out cross-validated Mahalanobis distance 
    between every pair of conditions.  Expected value is zero when two conditions 
    are identical, making the estimator unbiased.

    Parameters
    ----------
    trialsX : ndarray, shape (N_ch, C, T, Tr)
        Single-trial data returned by ``build_marg_arrays`` for one condition
        variable (or multiple variables flattened).  NaN-padded for conditions 
        with fewer than max trials.
    trialNum : ndarray, shape (N_ch, C)
        Per-condition trial counts (same convention as ``build_marg_arrays``).
    GO_CUE : int
        Index of the alignment event in the time dimension of trialsX.
    T_STA, T_END : int
        Start and end sample offsets in bins (relative to GO_CUE) defining the 
        analysis window.

    Returns
    -------
    crossnobis: ndarray, shape (C, C)
        Signed-sqrt of squared crossnobis distance matrix.
    """
    
    # slice window and transpose to (C, T_window, N_ch, Tr)
    Xcen = trialsX[:, :, GO_CUE + T_STA:GO_CUE + T_END].transpose(1, 2, 0, 3)

    # subtract condition-independent (direction-averaged, trial-averaged) mean
    # - has no effect on final result though (a mean shared across all conditions
    #   cancels in every pairwise difference and in the within-condition residuals)
    Xcen = Xcen - np.nanmean(Xcen, axis=(0, 3))[None, :, :, None]

    # time-average → (C, N_ch, Tr)
    Xcorr = np.nanmean(Xcen, axis=1)
    n_cond = Xcorr.shape[0]

    # get array of trial counts for each condition (C,)
    trialNum_flat = trialNum[0, :]

    # compute residuals for each condition (from mean)
    residuals = []
    for c in range(n_cond):
        tr_c = int(trialNum_flat[c])
        Xc = Xcorr[c, :, :tr_c].T  # (tr_c, N_ch)
        residuals.append(Xc - np.nanmean(Xc, axis=0))
    residuals = np.concatenate(residuals, axis=0)
    n_total = residuals.shape[0]

    # pooled within-condition noise covariance
    Sigma = residuals.T @ residuals / (n_total - n_cond)
    Sigma_inv = np.linalg.pinv(Sigma)

    # leave-one-out squared crossnobis for each condition pair (cond1, cond2)
    crossnobis = np.zeros((n_cond, n_cond))
    for cond1 in range(n_cond):
        for cond2 in range(n_cond):
            # get the number of trials for each condition and use the smaller of the two
            tr1 = int(trialNum_flat[cond1])
            tr2 = int(trialNum_flat[cond2])
            n = min(tr1, tr2)
            # trials x features (n, N_ch) arrays for each condition
            X1 = Xcorr[cond1, :, :n].T  
            X2 = Xcorr[cond2, :, :n].T
            # compute leave-one-out crossnobis estimates for each trial pair (n,)
            estimates = np.zeros(n)
            for k in range(n):
                # leave out trial k from both conditions and compute the distance between 
                # the means of the remaining trials
                train_idx = np.concatenate([np.arange(0, k), np.arange(k + 1, n)])
                d_train = np.nanmean(X1[train_idx], axis=0) - np.nanmean(X2[train_idx], axis=0)
                # compute the distance between the left-out trial pair (k)
                d_test = X1[k] - X2[k]
                # compute the cross-validated squared Mahalanobis distance for this trial pair
                estimates[k] = d_test @ Sigma_inv @ d_train
            # average the estimates across all trial pairs to get the final (squared) crossnobis 
            # distance for this condition pair
            crossnobis[cond1, cond2] = np.mean(estimates)

    # return signed-sqrt crossnobis distance matrix (preserves the sign of the original squared distance)
    return np.sign(crossnobis) * np.sqrt(np.abs(crossnobis))


def decoder_sweep_parallel(X, y, sorted_idxs, i, cv_splits):
    """Fit and evaluate one CV split of a linear SVM decoder/classifier.

    Intended for parallel execution via ``joblib.Parallel``.

    Parameters
    ----------
    X : ndarray, shape (n_trials, T, N_ch)
        Neural features array (neural activity across time for each trial).
    y : ndarray, shape (n_trials,)
        Integer condition labels.
    sorted_idxs : ndarray, shape (n_trials,) 
        Ordered trial indices for generating cross-validation splits.
    i : int
        Index of the current cross-validation fold
    cv_splits : int
        Number of cross-validation folds (if n_trials == cv_splits, then it's 
        leave-one-out cross-validation)

    Returns
    -------
    acc_chance, acc_test : float
        Chance-level and test accuracy for this fold
    Y_test, Y_pred : ndarray
        True and predicted labels for this fold
    """

    print(f'split: {i}')

    # select the test and train indices for this fold
    # if cv_splits == sorted_idxs.shape[0] then this is leave-one-out cross-validation
    idxs_test = sorted_idxs[i::cv_splits]
    idxs_train = np.setdiff1d(sorted_idxs, idxs_test)

    # generate the training and test sets, time and feature dimensions are flattened 
    # for the SVM input
    X_train = X[idxs_train].reshape((len(idxs_train), -1))
    Y_train = y[idxs_train]
    X_test = X[idxs_test].reshape((len(idxs_test), -1))
    Y_test = y[idxs_test]

    # fit a linear SVM with inner 10-fold cross-validation to select the best regularization parameter C
    pipe = Pipeline([('decode', SVC(kernel='linear', cache_size=1000, class_weight='balanced'))])
    param_grid = {'decode__C': np.logspace(-3.5, -2, 7)}
    search = GridSearchCV(pipe, param_grid, scoring=make_scorer(accuracy_score), cv=10)
    search.fit(X_train, Y_train)

    # extract the best model and its regularization parameter C
    mdl = search.best_estimator_.named_steps['decode']
    C = mdl.C

    # compute training and test accuracy for this fold
    acc_train = accuracy_score(Y_train, mdl.predict(X_train))
    acc_test = accuracy_score(Y_test, mdl.predict(X_test))
    # compute the chance accuracy based on the majority class in the full dataset
    acc_chance = np.bincount(y.ravel()).max() / len(y.ravel())

    print(f'train acc: {acc_train} (best C: {C})')
    print(f'test acc: {acc_test}')
    print(f'chance acc: {acc_chance}')

    # compute the predicted labels for the test set
    Y_pred = mdl.predict(X_test)

    # return the chance accuracy, test accuracy, true labels, and predicted labels for this fold
    return acc_chance, acc_test, Y_test, Y_pred


def run_decoding_sweep(X_neural, y_labels, CONDITIONS, CONDITION_ARRAYS,
                       trial_mask, T_STA, T_END, n_jobs=32):
    """Run a parallel leave-one-out SVM decoding sweep and return aggregated results.

    Parameters
    ----------
    X_neural : ndarray, shape (n_trials, T, N_ch)
        Neural features array (neural activity across time for each trial).
    y_labels : ndarray, shape (n_trials,)
        Continuous condition variable (e.g. angle in radians) used to build
        integer class labels via ``np.unique(..., return_inverse=True)``.
    CONDITIONS : list of ndarray
        Unique condition values for each variable.
    CONDITION_ARRAYS : list of ndarray
        Per-trial condition values, one array per variable.
    trial_mask : ndarray, shape (n_trials,), bool
        Selects which trials to include.
    T_STA, T_END : int
        Sample indices (for the time axis of X_neural) defining the decoding window.
    n_jobs : int
        Number of parallel workers for joblib.

    Returns
    -------
    dict with keys:
        y : ndarray, shape (n_trials,)
            Integer condition labels.
        true_all : ndarray, shape (n_trials,)
            True labels for all trials (concatenated across folds).
        pred_all : ndarray, shape (n_trials,)
            Predicted labels for all trials (concatenated across folds).
        true_all_split : list of ndarray
            True labels for each fold.
        pred_all_split : list of ndarray
            Predicted labels for each fold.
        acc_chance_all : ndarray, shape (cv_splits,)
            Chance-level accuracy for each fold.
        acc_test_all : ndarray, shape (cv_splits,)  
            Test accuracy for each fold.
        sorted_idxs : ndarray, shape (n_trials,)
            Trial indices sorted by condition codes for stratified cross-validation.
    """

    # convert continuous condition labels to integer class labels
    y_int = np.expand_dims(np.unique(y_labels, return_inverse=True, axis=0)[1], axis=1).flatten()
    # create one-hot encoding of the integer labels
    Y = np.zeros((y_int.shape[0], np.unique(y_int).shape[0]))
    for i in range(np.unique(y_int).shape[0]):
        Y[y_int == i, i] = 1
    # apply the trial mask to select only the trials of interest
    Y = Y[trial_mask]
    # get the integer class labels for the selected trials
    y = np.argmax(Y, axis=1)

    # assign a unique condition code to each trial based on the combination of conditions
    cond_codes = np.full(Y.shape[0], np.nan)
    for i, cond_set in enumerate(itertools.product(*CONDITIONS)):
        mask = np.all([cond_set[j] == CONDITION_ARRAYS[j] for j in range(len(CONDITIONS))], axis=0)
        cond_codes[mask[trial_mask]] = i

    # sort the trial indices by condition codes
    idxs = np.arange(Y.shape[0])
    np.random.shuffle(idxs)
    sorted_idxs = idxs[np.argsort(cond_codes[idxs])]

    # cv_splits == Y.shape[0] means leave-one-out cross-validation
    cv_splits = Y.shape[0]

    # initialize arrays to hold results of cross-validation folds
    acc_chance_all = np.zeros(cv_splits)
    acc_test_all = np.zeros(cv_splits)
    true_all_split = []
    pred_all_split = []

    # slice the neural data to the selected trials and time window for decoding
    X = X_neural[trial_mask, T_STA:T_END, :]

    # run the decoding sweep in parallel across cross-validation folds
    with Parallel(n_jobs=n_jobs, require='sharedmem') as parallel:
        results = parallel(
            delayed(decoder_sweep_parallel)(X, y, sorted_idxs, i, cv_splits)
            for i in range(cv_splits)
        )

    # aggregate the results from each fold
    for i, (acc_chance, acc_test, Y_test, Y_pred) in enumerate(results):
        acc_chance_all[i] = acc_chance
        acc_test_all[i] = acc_test
        true_all_split.append(Y_test)
        pred_all_split.append(Y_pred)

    # return dict of aggregated results
    return dict(
        y=y,
        true_all=np.concatenate(true_all_split),
        pred_all=np.concatenate(pred_all_split),
        true_all_split=true_all_split,
        pred_all_split=pred_all_split,
        acc_chance_all=acc_chance_all,
        acc_test_all=acc_test_all,
        sorted_idxs=sorted_idxs,
    )


def extract_windowed_features(X_data, t_end_idxs, window_samp):
    """Obtain time-resolved moving windows of average neural activity.

    Parameters
    ----------
    X_data : ndarray, shape (n_trials, T, n_ch)
        Neural features array (neural activity across time for each trial).
    t_end_idxs : ndarray, shape (n_time,) 
        Last sample index of each moving window.
    window_samp : int
        Moving window length in samples.

    Returns
    -------
    ndarray, shape (n_time, n_trials, n_ch)
        Time-resolved moving windows of average neural activity, for each 
        trial and channel, ending at time n_time.
    """

    # compute the start indices of each moving window
    t_sta = t_end_idxs - window_samp + 1   
    # compute the mean neural activity within each moving window for each trial and channel
    # and stack the results into a single array of shape (n_time, n_trials, n_ch)
    return np.stack([X_data[:, s:e+1, :].mean(axis=1) for s, e in zip(t_sta, t_end_idxs)], axis=0)


def fit_svc(x_train, y_train, c_range):
    """Linear SVC with inner 10-fold cross-validation over c_range.

    Parameters
    ----------
    x_train : ndarray, shape (n_trials, n_features)
        Training data.
    y_train : ndarray, shape (n_trials,)
        Training labels.
    c_range : array-like
        Regularization values to search over

    Returns
    -------
    Fitted SVC (best estimator from GridSearchCV)
    """

    # fit a linear SVC with inner 10-fold cross-validation to select the best regularization parameter C
    pipe = Pipeline([('decode', SVC(kernel='linear', cache_size=1000, class_weight='balanced'))])
    search = GridSearchCV(pipe, {'decode__C': c_range}, scoring=make_scorer(accuracy_score), cv=10)
    search.fit(x_train, y_train)

    # return the best estimator (SVC) from the grid search
    return search.best_estimator_.named_steps['decode']


def cross_temporal_decoding(X_delay, X_move, y, sorted_idxs,
                            cv_splits=10, c_range=None, label=''):
    """Cross-temporal decoding across delay and move epochs.

    Trains a linear SVC at each delay time point, tests at all delay and move
    time points (and vice versa), yielding four accuracy matrices.

    Parameters
    ----------
    X_delay : ndarray, shape (n_delay_t, n_trials, n_ch)
        Time-resolved neural feature windows across the delay epoch.
    X_move  : ndarray, shape (n_move_t,  n_trials, n_ch)
        Time-resolved neural feature windows across the move epoch.
    y : ndarray, shape (n_trials,)
        Integer condition labels for each trial.
    sorted_idxs : ndarray
        Ordered trial indices for generating cross-validation splits.
    cv_splits : int
        Number of cross-validation folds (if n_trials == cv_splits, then it's
        leave-one-out cross-validation)
    c_range : array-like
        Regularization values to search over; defaults to logspace(-3.5, -1, 11)
    label : str
        Printed in progress messages (e.g. 'Direction')

    Returns
    -------
    acc_dd : (n_delay_t, n_delay_t, cv_splits) 
        Decoding accuracy matrix: trained on delay, tested ondelay
    acc_dm : (n_delay_t, n_move_t,  cv_splits)
        Decoding accuracy matrix: trained on delay, tested on move
    acc_md : (n_move_t,  n_delay_t, cv_splits)
        Decoding accuracy matrix: trained on move, tested on delay
    acc_mm : (n_move_t,  n_move_t,  cv_splits)
        Decoding accuracy matrix: trained on move, tested on move
    chance_empirical : float
        Full-data majority-class chance level
    """

    # set default regularization range if not provided
    if c_range is None:
        c_range = np.logspace(-3.5, -1, 11)

    # initialize accuracy matrices
    n_delay_t = X_delay.shape[0]
    n_move_t  = X_move.shape[0]
    acc_dd = np.zeros((n_delay_t, n_delay_t, cv_splits))
    acc_dm = np.zeros((n_delay_t, n_move_t,  cv_splits))
    acc_md = np.zeros((n_move_t,  n_delay_t, cv_splits))
    acc_mm = np.zeros((n_move_t,  n_move_t,  cv_splits))

    # compute the empirical chance level based on the majority class in the full dataset
    chance_empirical = np.bincount(y).max() / len(y)

    # build variables for printing progress messages
    tag       = f' ({label})' if label else ''
    t0        = time.time()
    n_trains  = cv_splits * (n_delay_t + n_move_t)
    trained   = 0

    # loop across cross-validation splits (trial folds)
    for split in range(cv_splits):

        # build the train/test split for this fold
        idxs_test  = sorted_idxs[split::cv_splits]
        idxs_train = np.setdiff1d(sorted_idxs, idxs_test)
        y_train, y_test = y[idxs_train], y[idxs_test]

        # print progress message for this fold
        print(f'CTD{tag} split {split+1}/{cv_splits}')

        # loop across delay time points
        for i_tr in range(n_delay_t):

            # fit the SVC on the training data for this delay time point
            mdl = fit_svc(X_delay[i_tr][idxs_train], y_train, c_range)

            # compute accuracy on all delay and move time points for this trained model
            for i_te in range(n_delay_t):
                acc_dd[i_tr, i_te, split] = accuracy_score(y_test, mdl.predict(X_delay[i_te][idxs_test]))
            for i_te in range(n_move_t):
                acc_dm[i_tr, i_te, split] = accuracy_score(y_test, mdl.predict(X_move[i_te][idxs_test]))

            # progress log
            trained += 1
            elapsed = time.time() - t0
            print(f'  delay train {i_tr+1}/{n_delay_t} | elapsed {elapsed:.0f}s | ETA {elapsed/trained*(n_trains-trained):.0f}s')

        # loop across move time points
        for i_tr in range(n_move_t):

            # fit the SVC on the training data for this move time point
            mdl = fit_svc(X_move[i_tr][idxs_train], y_train, c_range)

            # compute accuracy on all delay and move time points for this trained model
            for i_te in range(n_move_t):
                acc_mm[i_tr, i_te, split] = accuracy_score(y_test, mdl.predict(X_move[i_te][idxs_test]))
            for i_te in range(n_delay_t):
                acc_md[i_tr, i_te, split] = accuracy_score(y_test, mdl.predict(X_delay[i_te][idxs_test]))

            # progress log
            trained += 1
            elapsed = time.time() - t0
            print(f'  move  train {i_tr+1}/{n_move_t} | elapsed {elapsed:.0f}s | ETA {elapsed/trained*(n_trains-trained):.0f}s')

    # print the total elapsed time for the cross-temporal decoding sweep
    print(f'CTD{tag} done — total {time.time()-t0:.0f}s')

    # return all accuracy matrices and the full-data chance level
    return acc_dd, acc_dm, acc_md, acc_mm, chance_empirical


def alignment_index(data1, data2, 
                    cond_ids1, cond_ids2, 
                    t_start, t_end,
                    t_start2=None, t_end2=None,
                    n_pcs=10, subtract_ci=True):
    """Alignment index between two neural datasets (Elsayed et al. 2016, https://www.nature.com/articles/ncomms13239).

    Quantifies the overlap between the top-PC subspaces of two datasets.
    Index = 0: subspaces orthogonal; index = 1: subspaces fully overlapping.

    Parameters
    ----------
    data1, data2 : ndarray, shape (trials, time, channels)
        Neural data arrays.
    cond_ids1, cond_ids2 : ndarray, shape (trials,)
        Condition labels for each trial.
    t_start, t_end : int
        Start (inclusive) and end (exclusive) time indices for data1.
    t_start2, t_end2 : int, optional
        Time indices for data2. If None, t_start/t_end are used.
    n_pcs : int
        Number of top PCs to use (default 10).
    subtract_ci : bool
        Forced True in code. Subtract the condition-independent mean before analysis 
        so data matrix is zero column-mean; required for the projection and eigenvalue
        normalization to be consistent (a False value is overridden below).

    Returns
    -------
    ai_1to2, ai_2to1 : float
        Variance of one dataset X captured by the other dataset Y's PCs (ai_XtoY), 
        normalised by the variance it captures in its own PCs (directional alignment 
        indices).
    ai_mean : float
        Mean of the two above alignment indices.
    vaf_1to2, vaf_2to1 : ndarray, shape (n_pcs,)
        Per-PC contributions to ai_1to2 / ai_2to1.
    D1, D2 : ndarray, shape (channels, n_pcs)
        Top PCs of each dataset (columns are PC vectors).
    scale1, scale2 : float
        Fraction of each dataset's total variance held in its top-k PCs; multiply
        returned vaf values by this scale to convert them from a top-k fraction to
        a fraction of total activity variance.
    """

    if t_start2 is None:
        t_start2 = t_start
    if t_end2 is None:
        t_end2 = t_end

    # forced True: projection/eigenvalue normalization only consistent when data matrix
    # is zero column-mean
    subtract_ci = True

    # slice analysis window - (trials, T, channels)
    d1 = data1[:, t_start:t_end, :]
    d2 = data2[:, t_start2:t_end2, :]

    # get unique condition labels for each dataset
    conds1 = np.unique(cond_ids1)
    conds2 = np.unique(cond_ids2)

    # condition-averaged activity - (n_conds, T, channels)
    ca1 = np.stack([d1[cond_ids1 == c].mean(axis=0) for c in conds1])
    ca2 = np.stack([d2[cond_ids2 == c].mean(axis=0) for c in conds2])

    # subtract condition-independent mean
    if subtract_ci:
        ca1 = ca1 - ca1.mean(axis=0, keepdims=True)
        ca2 = ca2 - ca2.mean(axis=0, keepdims=True)

    # reshape data matrices to (conditions*time, channels)
    n_chans = ca1.shape[2]
    P1 = ca1.transpose(2, 0, 1).reshape(n_chans, -1).T
    P2 = ca2.transpose(2, 0, 1).reshape(n_chans, -1).T

    # compute top-k PCs of each data matrix; singular_values_**2 are the eigenvalues of P.T @ P
    def _top_pcs(P, k):
        pca = PCA(n_components=k)
        pca.fit(P)
        # return the top-k PC vectors (columns) and their corresponding eigenvalues
        return pca.components_.T, pca.singular_values_ ** 2
    D1, ev1 = _top_pcs(P1, n_pcs)
    D2, ev2 = _top_pcs(P2, n_pcs)

    # project each dataset onto the other's subspace
    # data1 in data2's subspace
    proj1 = P1 @ D2  
    # data2 in data1's subspace
    proj2 = P2 @ D1

    # per-PC VAF (normalized by top-k eigenvalue sum)
    vaf_1to2 = (proj1 ** 2).sum(axis=0) / ev1.sum()
    vaf_2to1 = (proj2 ** 2).sum(axis=0) / ev2.sum()
    # sum VAF over PCs to get the alignment index
    ai_1to2 = vaf_1to2.sum()
    ai_2to1 = vaf_2to1.sum()

    # compute scale factors to convert vaf from top-k fraction to total-activity fraction
    scale1 = ev1.sum() / (P1 ** 2).sum()
    scale2 = ev2.sum() / (P2 ** 2).sum()

    return (ai_1to2, ai_2to1, 
            (ai_1to2 + ai_2to1) / 2,
            vaf_1to2, vaf_2to1, D1, D2,
            scale1, scale2)


def cv_distance(X1, X2, subtract_mean=False):
    """Cross-validated (unbiased) distance between two distributions.

    Adapted from Python code by Benyamin Meschede-Krasa, based on the MATLAB code
    at https://github.com/fwillett/cvVectorStats/blob/master/cvDistance.m
    
    Parameters
    ----------
    X1, X2 : ndarray, shape (nTrials, nFeatures)
        Trial data for each distribution. Must have the same shape.
    subtract_mean : bool
        Center each mean-difference vector before the dot product.

    Returns
    -------
    sq_dist, euc_dist : float
        Squared and signed-sqrt cross-validated distance.
    """

    # convert inputs to numpy arrays and check that they have the same shape
    X1 = np.array(X1)
    X2 = np.array(X2)
    assert X1.shape == X2.shape, "Distributions must have same shape"

    # container for the cross-validated distance estimates for each trial set
    n_trials, _ = X1.shape
    sq_dist_est = np.zeros([n_trials, 1])

    # loop over trials, leaving one out each time to compute the cross-validated distance
    for tr in range(n_trials):      
        # create indices for the "big" set (all trials except the left-out one) 
        # and the "small" set (the left-out trial)
        big_set_idx = list(range(n_trials))
        small_set_idx = big_set_idx.pop(tr)
        # compute the mean difference between the two distributions for the big set and the small set
        mean_diff_big_set = np.mean(X1[big_set_idx, :] - X2[big_set_idx, :], axis=0)
        mean_diff_small_set = X1[small_set_idx, :] - X2[small_set_idx, :]
        # compute the cross-validated squared distance estimate for this trial set
        if subtract_mean:
            sq_dist_est[tr] = np.dot(mean_diff_big_set - np.mean(mean_diff_big_set),
                                 (mean_diff_small_set - np.mean(mean_diff_small_set)).transpose())
        else:
            sq_dist_est[tr] = np.dot(mean_diff_big_set, mean_diff_small_set.transpose())

    # average the squared distance estimates across all trial sets to get the final 
    # cross-validated squared distance
    sq_dist = np.mean(sq_dist_est)
    # compute the signed-sqrt cross-validated matrix (preserves the sign of the original squared distance)
    euc_dist = np.sign(sq_dist) * np.sqrt(np.abs(sq_dist))

    return sq_dist, euc_dist



