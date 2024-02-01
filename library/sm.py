import autograd.numpy as np

def timesm_obj(par, Ef, Ephis, tseq, model_t, model_tt, g, dg):

    dthetat  = model_t(par, tseq)
    d2thetat = model_tt(par, tseq)

    t1 = (Ephis * dthetat[:, :, None] * dthetat[:, None, :] * g[:, None, None]).sum((1,2))
    t2 = ((dg[:, None] * dthetat + g[:, None]*d2thetat) * Ef).sum(1)

    return (t1 + 2*t2).mean()

def l1_penalty(par, tseq, model_t):
    return np.abs(model_t(par, tseq)).mean()