from braindecode import EEGClassifier
from braindecode.models import EEGNet
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping
from sklearn.base import clone
import numpy as np
import torch

def save_eegnet_model(filepath, model, ch_names, template, **metadata):
    net = model if isinstance(model, torch.nn.Module) else getattr(model, "module_", None)
    checkpoint = {**metadata, "ch_names": list(ch_names), "state_dict": net.state_dict(),
                  "n_chans":template["X"].shape[1],"n_times":template["X"].shape[2],}
    torch.save(checkpoint, filepath)

def load_eegnet_model(filepath):
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=True,)

    X_val = np.empty((0, checkpoint["n_chans"], checkpoint["n_times"]),dtype=np.float32)
    y_val = np.empty(0, dtype=np.int64)

    model_kwargs = dict(checkpoint["model_kwargs"])
    model_kwargs.setdefault("classes", [0, 1])

    clf = set_eegnet_model(X_val, y_val, **model_kwargs)
    clf.initialize()
    clf.module_.load_state_dict(checkpoint["state_dict"])
    clf.module_.eval()

    return clf, checkpoint

def set_eegnet_model(X_val, y_val, sfreq=128, **kwargs):
    if len(y_val) > 0:
        train_split = predefined_split(Dataset(X_val, y_val))
        callbacks = [("early_stopping", EarlyStopping( 
            monitor="valid_loss", patience=10, lower_is_better=True,load_best=True))]
    else:
        train_split = None
        callbacks = []
    model = EEGClassifier(
        module=EEGNet,
        module__n_chans=X_val.shape[1],
        module__n_times=X_val.shape[2],
        module__n_outputs=2,
        module__sfreq=sfreq,
        #module__kernel_length=int(sfreq / 2),
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=train_split,
        callbacks=callbacks,
        iterator_train__shuffle=True,
        iterator_train__drop_last=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=0,
    )
    model.set_params(**kwargs)
    return model
    

def adapt_eegnet_head(clf, X_cal, y_cal, **kwargs):
    raise NotImplementedError("get it later")

def adapt_eegnet_full(source_model, X_cal, y_cal, **kwargs):
    clf = clone(source_model)
    clf.set_params(**kwargs)
    clf.set_params(train_split=None, callbacks=[])

    clf._set_signal_args(X_cal, y_cal, classes=None)
    clf.signal_args_set_ = True

    clf.initialize()
    clf.module_.load_state_dict(source_model.module_.state_dict())

    for param in clf.module_.parameters():
        param.requires_grad_(True)

    clf.partial_fit(X_cal, y_cal)
    return clf