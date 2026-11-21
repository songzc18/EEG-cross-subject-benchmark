import torch
import numpy as np
import pandas as pd
import fm_evaluation.modules.utils as utils
from pathlib import Path
from braindecode.util import set_random_seeds
from .models import make_models, adapt_models, save_models, load_models

def check_source_model(source_model, source_model_path, model, ch_names, **datasets):
    if source_model_path.exists():
        source_model, checkpoint = load_models[model](source_model_path)
        ch_names = list(checkpoint["ch_names"])
        for dataset in datasets.values():
            for key, data in dataset.items():
                dataset[key] = utils.check_channels(data,ch_names)
    return source_model, ch_names

def train_self(model, train_, val_, protocol_, kwargs, source_model, **_unused):
    clf = make_models[model](val_["X"], val_["y"], **kwargs)
    clf.fit(protocol_["X"], protocol_["y"])
    return clf

def train_loso(model, train_, val_, protocol_, kwargs, source_model, **_unused):
    if source_model is None:
        clf = make_models[model](val_["X"], val_["y"], **kwargs)
        clf.fit(train_["X"], train_["y"])
        return clf
    return source_model

def train_pop(model, train_, val_, protocol_, kwargs, source_model, **_unused):
    if source_model is None:
        clf = make_models[model](val_["X"], val_["y"], **kwargs)
        clf.fit(protocol_["X"], protocol_["y"])
        return clf
    return source_model

def train_cal(model, train_, val_, protocol_, kwargs, source_model, ft_model, ft_kwargs):
    if source_model is None:
        raise ValueError("Prepare the LOSO source model before running CAL")
    return adapt_models[ft_model](source_model, protocol_["X"], protocol_["y"], **ft_kwargs)

train_models = { "SELF": train_self, "POP": train_pop, "LOSO": train_loso, "CAL": train_cal}

def predict_fold_rows(clf, test_dataset, row_info):
    rows = []
    right_col = int(np.flatnonzero(np.asarray(clf.classes_) == 1)[0])
    for key, data in test_dataset.items():
        sub, ses, phase = key
        y_test = data["y"]
        y_pred = clf.predict(data["X"])
        p_right = clf.predict_proba(data["X"])[:, right_col]
        for idx, true, pred, prob in zip(data["trial_idx"], y_test, y_pred, p_right):
            rows.append({
                **row_info,
                "test_idx": int(idx),
                "test_ses": ses,
                "test_phase": phase,
                "y_true": int(true),
                "y_pred": int(pred),
                "p_right": float(prob),
            })
    return rows

def evaluate_cv(protocol, filepath, sub, model, train_sess=(1,), test_sess=(1, 2),
                train_phases=("offline",), test_phases=("offline", "online"), val_size=0, 
                k_values=(0, 8, 20, 40, 60, 80), n_splits=5, seed=0, folds=(0,), 
                protocol_sess=(1,), protocol_phases=("offline",), protocol_seeds=(0,), 
                subjects=np.arange(1, 55), model_kwargs=None, *, ft_model=None, ft_kwargs=None, modelpath=None):
    # Parameter Initialization
    protocol = protocol.upper()
    k_values = (0,) if protocol == "LOSO" else tuple(k for k in k_values if k != 0)
    model_kwargs = {} if model_kwargs is None else model_kwargs
    ft_kwargs = {} if ft_kwargs is None else ft_kwargs
    phase_order = ("offline", "online")
    train_phases = tuple(p for p in phase_order if p in train_phases)
    protocol_phases = tuple(p for p in phase_order if p in protocol_phases)
    test_phases = tuple(p for p in phase_order if p in test_phases)
    train_sess = tuple(sorted(set(train_sess)))
    test_sess = tuple(sorted(set(test_sess)))
    protocol_sess = tuple(sorted(set(protocol_sess)))
    train_sess_name = "-".join(map(str, train_sess))
    train_phases_name = "off-on" if len(train_phases) == 2 else train_phases[0]
    protocol_sess_name = "-".join(map(str, protocol_sess))
    protocol_phases_name = "off-on" if len(protocol_phases) == 2 else protocol_phases[0]
    protocol_seeds = (seed,) if protocol == "LOSO" else protocol_seeds
    # Get datasets and indices
    train_datasets, protocol_datasets, test_datasets, ch_names = utils.load_evaluation_data(
        filepath, protocol, sub, train_sess, train_phases, protocol_sess, protocol_phases, test_sess, test_phases, ch_names=None, subjects=subjects)
    train_indices, protocol_indices, test_indices = utils.make_cv_indices(train_datasets, protocol_datasets, test_datasets, seed, n_splits)
    train_indices, val_indices, protocol_indices = utils.prepare_val_indices(protocol, train_indices, protocol_indices, protocol_datasets, seed, val_size)
    protocol_indices = utils.prepare_protocol_indices(protocol_indices, protocol_datasets, seed, protocol_seeds)
    
    rows = []
    for fold in range(n_splits):
        source_model = None
        if folds is not None and fold not in folds:
            continue
        # Get flod datasets
        train_flod = utils.combine_fold_datasets(train_datasets, train_indices, fold)
        val_flod = utils.combine_fold_datasets(train_datasets, val_indices, fold) if protocol in {"LOSO", "CAL"} else utils.combine_fold_datasets(protocol_datasets, val_indices, fold)
        test_flod = utils.combine_fold_datasets(test_datasets, test_indices, fold)
        if protocol in {"LOSO", "CAL"}:
            filename = f"loso_{model}_sub-{sub:02d}_ses-{train_sess_name}_{train_phases_name}_val-{val_size}_seed-{seed}_fold-{fold}.pt"
            source_model_path =  Path(modelpath) / filename
            source_model, ch_names = check_source_model(source_model, source_model_path, model, ch_names,
                train_flod=train_flod, val_flod=val_flod, protocol_datasets=protocol_datasets,test_flod=test_flod)
        for protocol_seed in protocol_seeds:
            protocol_fold = utils.combine_fold_datasets(protocol_datasets, protocol_indices, fold, protocol_seed=protocol_seed)
            for kv in k_values:
                if protocol == "POP":
                    source_model = None
                    filename = f"pop_{model}_ses-{protocol_sess_name}_{protocol_phases_name}_val-{val_size}_seed-{seed}_fold-{fold}_k-{kv}_pseed-{protocol_seed}.pt"
                    source_model_path =  Path(modelpath) / filename
                    source_model, ch_names = check_source_model(source_model, source_model_path, model, ch_names,
                        train_flod=train_flod, val_flod=val_flod, protocol_fold=protocol_fold,test_flod=test_flod)
                # Get kv flod datasets
                template = next(iter(test_flod.values()))
                protocol_fold_k = utils.concatenate_datasets(protocol_fold, k=kv, template=template)
                train_flod_k = utils.concatenate_datasets(train_flod, template=template)
                val_flod_k = utils.concatenate_datasets(val_flod, template=template)
                
                set_random_seeds( int(seed), cuda=torch.cuda.is_available(), cudnn_benchmark=False)
                torch.backends.cudnn.deterministic = True
                clf = train_models[protocol](model, train_flod_k, val_flod_k, protocol_fold_k,
                        kwargs=model_kwargs, source_model=source_model, ft_model=ft_model, ft_kwargs=ft_kwargs)
                if protocol in {"LOSO", "POP"} and source_model is None:
                    save_models[model](source_model_path, clf, ch_names=ch_names, model_kwargs=model_kwargs, template=template)
                row_info = {
                    "decoder": model,
                    "protocol": protocol,
                    "subject": sub,
                    "seed": seed,
                    "fold": fold,
                    "train_sess": train_sess_name,
                    "train_phases": train_phases_name,
                    "protocol_sess": protocol_sess_name,
                    "protocol_phases": protocol_phases_name,
                    "n_target_train": kv,
                    "protocol_seed": protocol_seed,
                    "fine_tuned": ft_model if protocol == "CAL" else None,
                }
                rows.extend(predict_fold_rows(clf, test_flod, row_info))
    return pd.DataFrame(rows)