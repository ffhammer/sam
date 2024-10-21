import numpy as np
from scipy.interpolate import interp1d
from enum import Enum


LINEAR_INTER_STEPS = 10



def transform_none(conc, surv):
    return conc, surv


def transform_linear_interpolation_old(conc, surv):
    conc = conc.copy()
    surv = surv.copy()
    
    c0 = conc[1] / 2
    points = np.linspace(np.log10(c0), np.log10(conc[-1]), LINEAR_INTER_STEPS)
    
    conc[0] = c0
    
    interp_func = interp1d(np.log10(conc), surv)
    return 10 ** points, interp_func(points)


def transform_linear_interpolation(conc, surv):
    conc = conc.copy()
    surv = surv.copy()
    
    # change first conc value
    c0 = conc[1] / 2
    conc_inter = conc.copy()
    conc_inter[0] = c0
    
    #change first_surv
    surv_inter = surv.copy()
    surv_inter[0] = surv_inter[1]
    
    
    points = np.linspace(np.log10(c0), np.log10(conc[-1]), LINEAR_INTER_STEPS)
    
    interp_func = interp1d(np.log10(conc_inter), surv_inter)
    
    interpolated_concentration = 10 ** points
    interpolated_survival = interp_func(points)
    
    conc = np.concatenate((conc[:1], interpolated_concentration))
    surv = np.concatenate((surv[:1], interpolated_survival))
    
    return conc, surv


def transform_williams(conc, surv):
    vec = np.array(surv)
    count = np.ones_like(vec)
    steps = vec[:-1] - vec[1:]
    outlier = np.where(steps < 0)[0]

    while outlier.size > 0:
        
        index = outlier[0]
        
        if index + 1 >= len(vec):
            break
        
        # Averaging over the current and the next value
        weighted_avg = np.average([vec[index], vec[index + 1]], weights=[count[index], count[index + 1]])
        vec[index] = weighted_avg
        count[index] += count[index + 1]
    
        # Removing the next value after the current index
        vec = np.delete(vec, index + 1)
        count = np.delete(count, index + 1)
        
        steps = vec[:-1] - vec[1:]
        outlier = np.where(steps < 0)[0]

    # Replicating values based on their counts
    vec_f = np.repeat(vec, count.astype(int))
    return conc, vec_f


def transform_williams_and_linear_interpolation(conc, surv):
    conc_t, surv_t = transform_williams(conc, surv)
    return transform_linear_interpolation(conc_t, surv_t)

def transform_williams_and_linear_interpolation_old(conc, surv):
    conc_t, surv_t = transform_williams(conc, surv)
    return transform_linear_interpolation_old(conc_t, surv_t)


class Transforms(Enum):
    none = transform_none
    linear_interpolation = transform_linear_interpolation
    williams = transform_williams
    williams_and_linear_interpolation = transform_williams_and_linear_interpolation
    williams_and_linear_interpolation_old = transform_williams_and_linear_interpolation_old