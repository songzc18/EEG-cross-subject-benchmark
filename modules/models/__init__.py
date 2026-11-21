from . import eegnet_adapter, labram_adapter

make_models = {
    "eegnet": eegnet_adapter.set_eegnet_model,
    "labram": labram_adapter.set_labram_model,
}

adapt_models = {
    "eegnet_full": eegnet_adapter.adapt_eegnet_full,
    "eegnet_head": eegnet_adapter.adapt_eegnet_head,
    "labram_full": labram_adapter.adapt_labram_full,
}

save_models = {
    "eegnet": eegnet_adapter.save_eegnet_model,
    "labram": labram_adapter.save_labram_model,
}

load_models = {
    "eegnet": eegnet_adapter.load_eegnet_model,
    "labram": labram_adapter.load_labram_model,
}