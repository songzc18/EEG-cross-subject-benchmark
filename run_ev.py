from fm_evaluation.modules.evaluation import evaluate_cv
import torch,argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sub", type=int, required=True)
parser.add_argument("--train_ses", type=int, required=True)
parser.add_argument("--protocol_ses", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--train_phase", choices=("offline", "online", "offon"), required=True)
parser.add_argument("--protocol_phase", choices=("offline", "online", "offon"), required=True)
parser.add_argument("--k_values", type=int, nargs="+", default=list(range(4, 81, 4)),)
parser.add_argument("--protocol", type=str.upper, choices=("SELF", "LOSO", "CAL", "POP"), required=True,)

args = parser.parse_args()
torch.set_num_threads(4)

filepath = "fm_evaluation/data/processed_8-30_128"
modelpath = "fm_evaluation/data/models"
model = "eegnet"
test_sess = (1,2)
test_phases = ("offline","online")
n_splits = 5
folds = None
k_values=(0,) if args.protocol == "LOSO" else tuple(args.k_values)
protocol_seeds = tuple(range(5))
val_size = 0

phase_options = {
    "offline": ("offline",),
    "online": ("online",),
    "offon": ("offline", "online"),
}

train_phases = phase_options[args.train_phase]
protocol_phases = phase_options[args.protocol_phase]

model_kwargs = {
    "sfreq": 128,
    "max_epochs": 64,
    "lr": 0.001,
    "batch_size": 16,
    "module__kernel_length": 64,
    "module__drop_prob": 0.25,
}

ft_model="eegnet_full"

ft_kwargs = {
    "lr": 1e-4,
    "max_epochs": 20,
    "batch_size": 16,
}


out_data = evaluate_cv(
    protocol=args.protocol,
    filepath=filepath,
    sub=args.sub,
    model=model,
    train_sess=(args.train_ses,),
    train_phases=train_phases,#("offline", "online"),#
    test_sess=test_sess,
    test_phases=test_phases,
    val_size=val_size,
    k_values=k_values,
    n_splits=n_splits,
    folds=folds,
    protocol_sess=(args.protocol_ses,),
    protocol_phases=protocol_phases,#("offline", "online"),#
    protocol_seeds=protocol_seeds,
    seed=args.seed,
    model_kwargs=model_kwargs,
    ft_model=ft_model,
    ft_kwargs=ft_kwargs,
    modelpath=modelpath,
)

k_name = "-".join(map(str, k_values))
savepath = (
    f"fm_evaluation/data/result/{model}_8_30_128_v2/"
    f"{args.protocol.lower()}_sub-{args.sub:02d}"
    f"_train-{args.train_ses}_{args.train_phase}"
    f"_protocol-{args.protocol_ses}_{args.protocol_phase}"
    f"_k-{k_name}_seed-{args.seed}.csv"
)
out_data.to_csv(savepath, index=False)