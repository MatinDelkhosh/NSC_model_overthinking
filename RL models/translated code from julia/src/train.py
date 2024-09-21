import torch

def gmap(f, prms, *gss):
    # Initialize gradients dictionary
    gsout = {p: torch.zeros_like(p) for p in prms}
    return gmap_(f, gsout, *gss)

def gmap_(f, gsout, *gss):
    for ip, p in enumerate(gsout.keys()):
        # Extract gradients for the current parameter from all gradient dictionaries
        gs_values = [gss_i.get(p, torch.zeros_like(p)) for gss_i in gss]
        # Apply function f to the gradients and store result
        gsout[p] = f(*gs_values)
    return gsout

# Helper function to get a gradient or zero if it's missing
def _getformap(gs, p):
    g = gs.get(p, None)
    if g is None:
        return torch.zeros_like(p)
    return g