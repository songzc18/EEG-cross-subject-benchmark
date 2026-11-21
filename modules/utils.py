import pickle
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def load_data(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

def check_channels(data, reference_ch_names):
    if data["ch_names"] != reference_ch_names:
        order = [data["ch_names"].index(ch) for ch in reference_ch_names]
        data["X"] = data["X"][:, order, :]
        data["ch_names"] = list(reference_ch_names)
    return data
    
def load_evaluation_data(filepath, protocol, sub, train_sess, train_phases, protocol_sess, protocol_phases, test_sess, test_phases, ch_names=None, subjects=np.arange(1,55)):
    filepath = Path(filepath)
    cache = {}
    # define infunction
    def get_data(subject, ses, phase):
        nonlocal ch_names
        key = (int(subject), int(ses), phase)
        if key not in cache:
            data = load_data(filepath / f"sub-{subject:02d}_ses-{ses}_{phase}.pkl")
            if ch_names is None:
                ch_names = list(data["ch_names"])
            cache[key] = check_channels(data, ch_names)
        return cache[key]
    # set train dataset subjects
    if protocol == "POP":
        train_subjects = []
    elif protocol == "SELF": 
        train_subjects = [sub]
    else:
        train_subjects = np.setdiff1d(subjects, [sub])
    # set protocol dataset subect
    protocol_subjects = list(subjects) if protocol == "POP" else [sub]
    # load all datasets
    train_datasets = {(subject, ses, phase): get_data(subject, ses, phase)
        for subject in train_subjects for ses in train_sess for phase in train_phases }
    protocol_datasets = {(subject, ses, phase): get_data(subject, ses, phase)
        for subject in protocol_subjects for ses in protocol_sess for phase in protocol_phases }
    test_datasets = {(sub, ses, phase): get_data(sub, ses, phase)
        for ses in test_sess for phase in test_phases }
    return train_datasets, protocol_datasets, test_datasets, ch_names

def make_cv_indices(train_datasets, protocol_datasets, test_datasets, seed, n_splits):
    datasets = {**train_datasets, **protocol_datasets, **test_datasets}
    train_indices, protocol_indices, test_indices = {}, {}, {}
    for key, data in datasets.items():
        y = np.asarray(data["y"])
        splitter = StratifiedKFold( n_splits=n_splits, shuffle=True, random_state=int(seed))
        splits = list(splitter.split(np.zeros(len(y)), y))
        if key in train_datasets:
            train_indices[key] = np.stack([train_idx for train_idx, _ in splits])
        if key in test_datasets:
            test_indices[key] = np.stack([test_idx for _, test_idx in splits])
        if key in protocol_datasets:
            protocol_indices[key] = np.stack([protocol_idx for protocol_idx, _ in splits])
    return train_indices, protocol_indices, test_indices

def prepare_val_indices(protocol, train_indices, protocol_indices, protocol_datasets, seed, val_size):
    train_result, val_result, protocol_result  = {}, {}, {}
    if protocol in {"LOSO", "CAL"}:
        subjects = sorted({key[0] for key in train_indices})
        rng = np.random.default_rng(int(seed))
        val_subjects = rng.choice(subjects, size=val_size, replace=False)
        for key, data in train_indices.items():
            result = val_result if key[0] in val_subjects else train_result
            result[key] = data
        protocol_result = protocol_indices
    elif protocol in {"SELF", "POP"}:
        for key, data in protocol_indices.items():
            sub, ses, phase = key
            protocol_result[key], val_result[key] = [], []
            for fold, fold_data in enumerate(data):
                rng = np.random.default_rng(np.random.SeedSequence([
                    int(seed), int(sub), int(ses), int(phase == "online"), fold]))
                y_fold = np.asarray(protocol_datasets[key]["y"])[fold_data]
                labels = np.unique(y_fold)
                val_pos = np.concatenate([rng.choice(np.flatnonzero(y_fold == label),
                            size=val_size // 2, replace=False) for label in labels])
                mask = np.ones(len(fold_data), dtype=bool)
                mask[val_pos] = False
                protocol_result[key].append(fold_data[mask])
                val_result[key].append(fold_data[val_pos])
            protocol_result[key] = np.stack(protocol_result[key])
            val_result[key] = np.stack(val_result[key])
        #train_result = train_indices
    return train_result, val_result, protocol_result

def prepare_protocol_indices(protocol_indices, protocol_datasets, seed, protocol_seeds):
    protocol_result = {}
    for key, data in protocol_indices.items():
        sub, ses, phase = key
        protocol_result[key] = []
        for fold, protocol_idx in enumerate(data):
            y_fold = np.asarray(protocol_datasets[key]["y"])[protocol_idx]
            labels = np.unique(y_fold)
            protocol_orders = {}
            for protocol_seed in protocol_seeds:
                rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(protocol_seed), fold, int(sub), int(ses), int(phase == "online")]))
                class_indices = [rng.permutation( protocol_idx[y_fold == label] ) for label in labels]
                protocol_orders[protocol_seed] = np.column_stack(class_indices).ravel()
            protocol_result[key].append(protocol_orders)
    return protocol_result

def combine_fold_datasets(datasets, indices, fold, protocol_seed=None):
    out = {}
    for key, data in indices.items():
        idx = data[fold][protocol_seed] if protocol_seed is not None else data[fold]
        out[key] = {"X": datasets[key]["X"][idx], "y": datasets[key]["y"][idx],
                    "ch_names": datasets[key]["ch_names"], "trial_idx": idx}
    return out

def concatenate_datasets(datasets, k=None, template=None):
    if not datasets:
        return {"X": template["X"][:0],
                "y": template["y"][:0]}
    return {name: np.concatenate([
        data[name][:k] for data in datasets.values()], axis=0)
        for name in ("X", "y")}
    
def summarize(results):
    group_cols = [
        "decoder", "protocol", "subject", "seed", "fold",
        "train_sess", "train_phases", "protocol_sess", "protocol_phases",
        "n_target_train", "protocol_seed", "fine_tuned", "test_ses", "test_phase",
    ]
    metric_cols = ["bacc", "kappa", "auc"]

    rows = []
    for keys, group in results.groupby(group_cols, sort=False, dropna=False):
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        p_right = group["p_right"].to_numpy()

        row = dict(zip(group_cols, keys))
        row.update(
            bacc=metrics.balanced_accuracy_score(y_true, y_pred),
            kappa=metrics.cohen_kappa_score(y_true, y_pred),
            auc=(
                metrics.roc_auc_score(y_true, p_right)
                if len(np.unique(y_true)) == 2
                else np.nan
            ),
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=group_cols + metric_cols)