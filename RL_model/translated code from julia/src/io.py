import pickle
from src.ToPlanOrNotToPlan import *

def recover_model(filename):
    # Load the model components
    with open(filename + "_opt.pkl", 'rb') as f:
        opt = pickle.load(f)

    with open(filename + "_hps.pkl", 'rb') as f:
        hps = pickle.load(f)

    with open(filename + "_progress.pkl", 'rb') as f:
        store = pickle.load(f)

    with open(filename + "_mod.pkl", 'rb') as f:
        network = pickle.load(f)

    with open(filename + "_policy.pkl", 'rb') as f:
        policy = pickle.load(f)

    with open(filename + "_prediction.pkl", 'rb') as f:
        prediction = pickle.load(f)

    return network, opt, store, hps, policy, prediction

def save_model(m, store, opt, filename, environment, loss_hp, Lplan):
    model_properties = m.model_properties
    network = m.network
    hps = {
        "Nhidden": model_properties.Nhidden,
        "T": environment.dimensions.T,
        "Larena": environment.dimensions.Larena,
        "Nin": model_properties.Nin,
        "Nout": model_properties.Nout,
        "GRUind": ToPlanOrNotToPlan.GRUind,
        "βp": loss_hp.βp,
        "βe": loss_hp.βe,
        "βr": loss_hp.βr,
        "Lplan": Lplan,
    }

    # Save components using pickle
    with open(filename + "_progress.pkl", 'wb') as f:
        pickle.dump(store, f)

    with open(filename + "_mod.pkl", 'wb') as f:
        pickle.dump(network, f)

    with open(filename + "_opt.pkl", 'wb') as f:
        pickle.dump(opt, f)

    with open(filename + "_hps.pkl", 'wb') as f:
        pickle.dump(hps, f)

    # Conditionally save additional fields
    if hasattr(m, 'policy'):
        with open(filename + "_policy.pkl", 'wb') as f:
            pickle.dump(m.policy, f)

    if hasattr(m, 'prediction'):
        with open(filename + "_prediction.pkl", 'wb') as f:
            pickle.dump(m.prediction, f)

    if hasattr(m, 'prediction_state'):
        with open(filename + "_prediction_state.pkl", 'wb') as f:
            pickle.dump(m.prediction_state, f)