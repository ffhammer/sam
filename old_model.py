import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import beta

def ecxsys(
    concentration : np.ndarray,
    hormesis_concentration : float,
    survival_tox_observed : np.ndarray,
    survival_max=100,
    curves_concentration_max=None,
    p=3.2,
    q=3.2,
):
    output = {"args":{
        "survival_max":100,
        "curves_concentration_max":None,
        "p":3.2,
        "q":3.2,
        "concentration": concentration,
        "hormesis_concentration" : hormesis_concentration,
        "survival_tox_observed": survival_tox_observed,
    }}

    if survival_max <= 0:
        raise ValueError("survival_max must be >= 0")
    if len(concentration) != len(survival_tox_observed):
        raise ValueError("concentration and survival_tox_observed must have the same length.")
    if len(concentration) > len(set(concentration)):
        raise ValueError("Concentrations must be unique.")
    if np.isnan(concentration).any():
        raise ValueError("Concentrations must be non None")
    if np.isnan(hormesis_concentration).any():
        raise ValueError("Concentrations must be non None")
    
    
    if hormesis_concentration not in concentration:
        raise ValueError("hormesis_concentration must equal one of the concentration values.")
    

    hormesis_index = np.where(hormesis_concentration == concentration)[0][0]
    if hormesis_index <= 1 or hormesis_index >= len(concentration):
        raise ValueError("hormesis_concentration must be greater than the lowest non-control concentration and less than the highest concentration")
        

    observations = survival_tox_observed

    if any(observations > survival_max) or any(observations < 0):
        raise ValueError("Observed survival must be between 0 and survival_max.")
    
    if len(set(concentration)) != len(concentration):
        raise ValueError("The values must be sorted by increasing concentration.")
    if any(np.array(concentration) < 0):
        raise ValueError("Concentrations must be >= 0")
    if min(concentration) > 0:
        raise ValueError("No control is given. The first concentration must be 0.")
    

    concentration = concentration.astype(np.float64)
    survival_tox_observed = survival_tox_observed.astype(np.float64)


    # we approximate a small control concentration?
    min_conc = 10 ** np.floor(np.log10(concentration[1]) - 2)
    concentration[0] = min_conc
    
    survival_tox_observed /= survival_max
    
    output.update(fit_LL5_model(min_conc=min_conc, survival_observed=survival_tox_observed, concentration=concentration))
    
    n_new = 3
    concentration = interpolate_sub_horm(concentration, hormesis_index - 1, hormesis_index, n_new, logarithmic=True)
    survival_tox_observed = interpolate_sub_horm(survival_tox_observed, hormesis_index - 1, hormesis_index, n_new, logarithmic=False)
    
    hormesis_index += n_new
    keep = np.setdiff1d(np.arange(len(concentration)), np.arange(hormesis_index - n_new, hormesis_index))
    
    survival_tox = np.concatenate(([1], survival_tox_observed[1:]))
    
    # drop sub hormesis
    survival_no_sub = np.concatenate((survival_tox[:2], survival_tox[hormesis_index:]))
    concentration_no_sub = np.concatenate((concentration[:2], concentration[hormesis_index:]))
    
    fitted_weibull = fit_weibull_model(survival_no_sub, concentration_no_sub)
    
    predicted_survival = fitted_weibull(concentration)
    
    output["fitted_weibull"] = fitted_weibull
    output["survival_pred"] = predicted_survival[keep] * survival_max
    
    
    output["stress"] = survival_to_stress(predicted_survival, p = p, q = q)[keep]
    
    
    
    len_curves = 1000
    conc_adjust_factor = 10 ** -5
    output['conc_adjust_factor'] = conc_adjust_factor
    if curves_concentration_max is None:
        curves_concentration_max = max(concentration)
    elif curves_concentration_max < min(concentration[concentration > 0]):
        raise ValueError("'curves_concentration_max' is too low.")
    
    
    curves_concentration = 10 ** np.linspace(np.log10(min_conc * conc_adjust_factor), np.log10(curves_concentration_max), len_curves)
    output['curves'] = predict_ecxsys(output, curves_concentration)

    conc_axis = make_axis_concentrations(curves_concentration, min_conc, conc_adjust_factor)
    output['curves']['concentration_for_plots'] = conc_axis['concentration']
    output['axis_break_conc'] = conc_axis['axis_break_conc']

    
    return output
    
def predict_ecxsys(model, concentration):
    
    p = model["args"]["p"]
    q = model["args"]["q"]
    survival_max = model["args"]["survival_max"]
    
    output = {"concentration":concentration}
    
    output["survival_LL5"] = model["survival_LL5_mod"](concentration) * survival_max
    
    pred_survival = model["fitted_weibull"](concentration)
    output["survival"] = pred_survival * survival_max
    
    output["stress"] = survival_to_stress(pred_survival, p=p, q=q)
    
    return output
    
    
def make_axis_concentrations(concentration, min_conc, conc_adjust_factor):
    use_for_plotting = (concentration < min_conc * conc_adjust_factor * 1.5) | (concentration > min_conc * 1.5)
    gap_idx = min(np.where(~use_for_plotting)[0])
    axis_break_conc = concentration[use_for_plotting][gap_idx]
    concentration[~use_for_plotting] = np.nan
    concentration[:gap_idx] /= conc_adjust_factor

    return {'concentration': concentration, 'axis_break_conc': axis_break_conc}

        

    
def fit_LL5_model(min_conc, concentration, survival_observed):

    conc_with_control_shifted = np.concatenate(([min_conc],concentration[1:]))
    survival_observed_averaged = moving_weighted_mean(survival_observed)
    
    
    interpolated = interp1d(np.log10(conc_with_control_shifted), survival_observed_averaged, kind='linear')
    conc_interpolated = 10 ** np.linspace(np.log10(min_conc), np.log10(concentration[-1]), 10)
    survival_interpolated = interpolated(np.log10(conc_interpolated))

    
    def fixed_LL5(x, b, e, f):
        c = 0
        d = survival_observed_averaged[0]
        
        if e <= 0.0:
            print("e is false")
        if 0 in x:
            print("here")
        
        denominator = (1 + np.exp(b * (np.log(x) - np.log(e)))) ** f
        
        return c + (d-c) / denominator
    
    
    bounds = ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
    
    popt, _ = curve_fit(fixed_LL5, conc_interpolated, survival_interpolated, bounds=bounds)
    survival_LL5_mod = lambda x: fixed_LL5(x, *popt)
    
    survival_LL5 = survival_LL5_mod(concentration)
    

    return {
        'survival_LL5_mod': survival_LL5_mod,
        'survival_LL5': survival_LL5
    }
    
def moving_weighted_mean(x):
    count = np.ones(len(x))
    x_diff = -np.diff(x)
    while any(x_diff < 0):
        i = np.where(x_diff < 0)[0][0]
        j = i + 1
        x[i] = np.average([x[i], x[j]], weights=[count[i], count[j]])
        x = np.delete(x, j)
        count[i] += count[j]
        count = np.delete(count, j)
        x_diff = -np.diff(x)
    return np.repeat(x, count.astype(int))


def interpolate_sub_horm(x, from_index, to_index, n_new, logarithmic=False):
    if logarithmic:
        x_new = np.logspace(np.log10(x[from_index]), np.log10(x[to_index]), n_new + 2)
    else:
        x_new = np.linspace(x[from_index], x[to_index], n_new + 2)
    return np.concatenate([x[:from_index], x_new, x[to_index +1:]])


def fit_weibull_model(survival, concentration):
    def weibull(x, b , e):
        
        return np.exp(-np.exp(b * (np.log(x) - e)))
    
    popt, _ = curve_fit(weibull, concentration, survival)
    return lambda x: weibull(x, *popt)


def clamp(x, lower=0, upper=1):
    """
    Clamps the values in x to be within the interval [lower, upper].
    """
    return np.minimum(np.maximum(x, lower), upper)

def survival_to_stress(survival, p=3.2, q=3.2):
    """
    Converts survival rates to stress values using the beta distribution CDF.
    
    Parameters:
    - survival: array-like, survival rates
    - p: shape parameter p of the beta distribution
    - q: shape parameter q of the beta distribution
    
    Returns:
    - stress: array-like, stress values
    """
    if p < 0 or q < 0:
        raise ValueError("Parameters p and q must be non-negative.")
    
    survival = clamp(survival)
    return beta.ppf(1 - survival, p, q)