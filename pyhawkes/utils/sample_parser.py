import logging

import numpy as np

from pyhawkes.utils.param_db import ParamsDatabase
log = logging.getLogger("global_log")

def parse_sample(samples, s):
    """
    Convert sample s into a dictionary of parameters that can be consumed by 
    the model. This is a bit hacky since we have to hard code the translation
    between sample variable names and parameter names
    """
    model_params = ParamsDatabase()
    
    # Graph params
    model_params.addDatabase("graph_model")
    model_params["graph_model","A"] = np.squeeze(np.atleast_3d(samples["A_smpls"])[:,:,s].astype(np.bool))

    model_params["graph_model","A"] = np.atleast_2d(model_params["graph_model","A"]).copy(order="C")
    
    if "Y_smpls" in samples:
        model_params["graph_model","Y"] = np.squeeze(samples["Y_smpls"][:,s])
    if "A_tau_smpls" in samples:
        model_params["graph_model","tau"] = samples["A_tau_smpls"][0,s].astype(np.float)
    
    # Weight params
    model_params.addDatabase("weight_model")
    model_params["weight_model","W"] = np.squeeze(np.atleast_3d(samples["W_smpls"])[:,:,s].astype(np.float32))
    model_params["weight_model","W"] = np.atleast_2d(model_params["weight_model","W"]).copy(order="C")
    
    if "mu_W_smpls" in samples:
        model_params["weight_model","mu_W"] = np.squeeze(samples["mu_W_smpls"][:,:,s])
        model_params["weight_model","mu_W"] = model_params["weight_model","mu_W"].copy(order="C")
    if "sig_W_smpls" in samples:
        model_params["weight_model","sig_W"] = np.squeeze(samples["sig_W_smpls"][:,:,s])
        model_params["weight_model","sig_W"] = model_params["weight_model","sig_W"].copy(order="C")
    
    # Background params
    model_params.addDatabase("bkgd_model")
    if "lam_homog_smpls" in samples:
        model_params["bkgd_model","lam_homog"] = np.squeeze(samples["lam_homog_smpls"][:,s])
    if "beta_smpls" in samples:
        model_params["bkgd_model","beta"] = np.squeeze(samples["beta_smpls"][:,:,s])
    if "pr_tod" in samples:
        model_params["bkgd_model","pr_tod"] = samples["pr_tod"]
    if "lam_knots" in samples:
        model_params["bkgd_model","knots"] = samples["lam_knots"]
    if "lam_shared_smpls" in samples:
        model_params["bkgd_model","lam_shared"] = samples["lam_shared_smpls"][:,s].astype(np.float32)
    if "lam_mu_smpls" in samples:
        model_params["bkgd_model","lam_mu"] = samples["lam_mu_smpls"][:,s].astype(np.float32)
    
    # Impulse params
    model_params.addDatabase("impulse_model")
    if "g_mu_smpls" in samples:
        if samples["g_mu_smpls"].shape[0] == 1:
            model_params["impulse_model","g_mu"] = np.float(np.squeeze(samples["g_mu_smpls"][0,s]))
        else:
            model_params["impulse_model","g_mu"] = np.squeeze(samples["g_mu_smpls"][:,:,s]).astype(np.float32)
                
    if "g_tau_smpls" in samples:
        if samples["g_tau_smpls"].shape[0] == 1:
            model_params["impulse_model","g_tau"] = np.float(np.squeeze(samples["g_tau_smpls"][0,s]))
        else:
            model_params["impulse_model","g_tau"] = np.squeeze(samples["g_tau_smpls"][:,:,s]).astype(np.float32)
    elif "Beta_smpls" in samples:
        Beta_shape = samples["Beta_smpls"].shape
        
        Beta_smpls = samples["Beta_smpls"]
        if np.ndim(samples["Beta_smpls"])<4:
            Beta_smpls = np.reshape(samples["Beta_smpls"],(Beta_shape[0],Beta_shape[1],Beta_shape[2],-1))
        
        
        model_params["impulse_model", "Beta"] = Beta_smpls[:,:,:,s]
    elif "G" in samples:
        model_params["impulse_model", "G"] = samples["G"].astype(np.float32)
        model_params["impulse_model","G"] = model_params["impulse_model","G"].copy(order="C") 
        model_params["impulse_model", "Gt"] = samples["Gt"].astype(np.float32)
        model_params["impulse_model","Gt"] = model_params["impulse_model","Gt"].copy(order="C")
    if "obasis" in samples:
        model_params["impulse_model","obasis"] = samples["obasis"].astype(np.float32)
    if "impulse_model" in samples:
        model_params["impulse_model","impulse_model"] = samples["impulse_model"]
    
    # Location model
    model_params.addDatabase("location1")
    if "L_smpls" in samples:
        model_params["location1","L"] = np.squeeze(samples["L_smpls"][:,:,s]).astype(np.float32)
    # Location model
    model_params.addDatabase("cluster1")
    if "Y_smpls" in samples:
        model_params["cluster1","Y"] = np.squeeze(samples["Y_smpls"][:,s])
    
    return model_params
