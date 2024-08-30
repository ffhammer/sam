import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import beta
from scipy.interpolate import interp1d
from py_lmcurve_ll5 import lmcurve_ll5, ll5Params
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from functools import partial



# Constants
BETA = 3.2
CONC0_MAX_DY = 5.0 / 100
CONC0_MIN_EXP = -100
LINEAR_INTER_STEPS = 10
CURVE_FITTING = 'scipy'  # 'scipy' or 'lmcurve'
TRANSFORM = "williams_and_linear_interpolation"  # 'transform_none', 'transform_linear_interpolation', 'transform_williams', 'transform_williams_and_linear_interpolation'

def ll5(conc, b, c, d, e, f):
    return c + (d - c) / (1 + (conc / e) ** b) ** f


def ll5_args(conc, args : ll5Params):

    return ll5(conc, args.b, args.c, args.d, args.e, args.f)

def ll5_inv(surv, b, c, d, e, f):
    return e * (((d - c) / (surv - c)) ** (1 / f) - 1) ** (1 / b)

def ll5_inv_args(surv, args : ll5Params):
    return ll5_inv(surv, args.b, args.c, args.d, args.e, args.f)
    



def fit_ll5(conc, surv, constrains : ll5Params) -> ll5Params:
    """Curve fitting function based on the chosen method"""
    
    
    if CURVE_FITTING == 'scipy':
        
        fixed_params = {k: v for k, v in constrains.__dict__.items() if v is not None}
        
        bounds = {"b": [0, 100], "c": [0, max(surv)], "d": [0, 2*max(surv)], "e": [0, max(conc)], "f": [0.1, 10]}

        keep = {k: v for k, v in bounds.items() if k not in fixed_params}
        
        bounds_tup = tuple(zip(*keep.values())) 
        
        
        def fitting_func(conc, *args):
                
            params = fixed_params.copy()
            params.update({k: v for k, v in zip(keep.keys(), args)})
            return ll5(conc,**params)
        
        
        popt, pcov = curve_fit(fitting_func, conc, surv, p0=np.ones_like(bounds_tup[0]), bounds=bounds_tup)
        
        params = fixed_params.copy()
        params.update({k: v for k, v in zip(keep.keys(), popt)})
        return ll5Params(**params)
    
    elif CURVE_FITTING == 'lmcurve':
    
        return lmcurve_ll5(conc, surv, **constrains.__dict__)
    else:
        raise ValueError("Unsupported curve fitting method")

def qbet(x):
    return beta.ppf(x, BETA, BETA)

def pbet(x):
    return beta.cdf(x, BETA, BETA)

# Linear interpolation and inverse linear interpolation
def linear_inv(vec_x, vec_y, y):
    interp_func = interp1d(vec_y, vec_x, fill_value="extrapolate")
    return interp_func(y)

# Finding null concentration
def find_c0(conc, pa : ll5Params, pb : ll5Params):
    
    
    conc0_max_exp = int(np.floor(np.log10(conc[1])))

    for i in range(conc0_max_exp, CONC0_MIN_EXP -1, -1):
        c0 = 10 ** i
        ya = ll5_args(c0, pa)
        yb = ll5_args(c0, pb)
        ysam = pa.d * (1 - pbet(qbet(1 - ya / pa.d) + qbet(1 - (pb.d / pa.d))))
        if all(np.abs((np.array([ya, yb, ysam]) - np.array([pa.d, pb.d, pb.d])) / np.array([pa.d, pb.d, pb.d])) < CONC0_MAX_DY):
            return c0
    return None

# Transformation functions
def transform_none(conc, surv):
    return conc, surv

def transform_linear_interpolation(conc, surv):
    c0 = conc[1] / 2
    e0 = surv[1]
    points = np.linspace(np.log10(c0), np.log10(conc[-1]), LINEAR_INTER_STEPS)
    
    conc[0] = c0
    
    interp_func = interp1d(np.log10(conc), surv)
    return 10 ** points, interp_func(points)


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

@dataclass
class PredictionData:
    concentration: np.ndarray = field(default_factory=lambda: np.array([]))
    survival_a: np.ndarray = field(default_factory=lambda: np.array([]))
    survival_b: np.ndarray = field(default_factory=lambda: np.array([]))
    survival_sam: np.ndarray = field(default_factory=lambda: np.array([]))
    survival_ea: np.ndarray = field(default_factory=lambda: np.array([]))
    survival_ca: np.ndarray = field(default_factory=lambda: np.array([]))
    lc_sam_conc: np.ndarray = field(default_factory=lambda: np.array([]))
    lc_sam_surv: np.ndarray = field(default_factory=lambda: np.array([]))
    stress_a : np.ndarray = field(default_factory=lambda: np.array([]))
    stress_b : np.ndarray = field(default_factory=lambda: np.array([]))

    def to_csv(self, filename: str):
        import pandas as pd
        
        max_length = max(len(getattr(self, field)) for field in self.__dataclass_fields__)
        
        def pad(arr):
            return np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan)
        
        data = {
            "Concentration": pad(self.concentration),
            "Survival_A": pad(self.survival_a),
            "Survival_B": pad(self.survival_b),
            "SAM": pad(self.survival_sam),
            "EA": pad(self.survival_ea),
            "CA": pad(self.survival_ca),
            "LC_SAM_Concentration": pad(self.lc_sam_conc),
            "LC_SAM_Survival": pad(self.lc_sam_surv),
            "Stress_A": pad(self.stress_a),
            "Stress_B": pad(self.stress_b),
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

def generate_predictions(name, conc, surv_a, surv_b, transform_name, f = None):

    transform_func = globals()[f'transform_{transform_name}']

    def get_properties(surv):
        conc_t, surv_t = transform_func(conc, surv)
        
        fixed_params = ll5Params(b=None, c=0, d=surv_t[0], e=None, f=f)
        
        fit_result : ll5Params = fit_ll5(conc_t, surv_t, fixed_params)
        
        lc_surv = fit_result.d / 100 * np.array([90, 50])
        lc_conc = ll5_inv(lc_surv, fit_result.b, fit_result.c, fit_result.d, fit_result.e, fit_result.f)
        
        return {
            'params': fit_result,
            'lc': {'conc': lc_conc, 'surv': lc_surv},
            'conc_t': conc_t, 'surv_t': surv_t
        }
        
    pa = get_properties(surv_a)
    pb = get_properties(surv_b)
    
    
    conc_0 = find_c0(conc, pa['params'], pb['params'])
    conc_n = max(conc)
    conc = np.array([i if i != 0 else conc_0 for i in conc ])

    conc_line = np.logspace(np.log10(conc_0), np.log10(conc_n), 100)
    

    surv_line_a = ll5_args(conc_line, pa['params'])
    surv_line_b = ll5_args(conc_line, pb['params'])

    stress_a = qbet(1 - surv_line_a / pa['params'].d)
    stress_b = qbet(1 - pb["params"].d / pa['params'].d)
    
    surv_line_sam = pa['params'].d * (1 - pbet(stress_a + stress_b))
    
    lc_sam_surv = pb['params'].d / 100 * np.array([90, 50])
    lc_sam_conc = [linear_inv(conc_line, surv_line_sam, s) for s in lc_sam_surv]

    # Effect Addition (EA)
    surv_line_ea = (pb['params'].d / pa['params'].d) * surv_line_a

    # Concentration Addition (CA)
    conc_env_ca = pa['params'].e * (((pa['params'].d / pb['params'].d) ** (1 / pa['params'].f) - 1) ** (1 / pa['params'].b))
    surv_line_ca = ll5(conc_line + conc_env_ca, pa['params'].b, 0, pa['params'].d, pa['params'].e, pa['params'].f)


    predictions = PredictionData(
            concentration=conc_line,
            survival_a=surv_line_a,
            survival_b=surv_line_b,
            survival_sam=surv_line_sam,
            survival_ea=surv_line_ea,
            survival_ca=surv_line_ca,
            lc_sam_conc=lc_sam_conc,
            lc_sam_surv=lc_sam_surv,
            stress_a=stress_a,
            stress_b=np.array([stress_b]),
        )
    return predictions


def plot_predictions(predictions, name, transform_name):
    plt.figure(figsize=(10, 5))
    plt.title(f'{name} - {transform_name}')
    plt.xlabel('Concentration')
    plt.ylabel('Survival')
    plt.xscale('log')

    # Unpacking and plotting
    plt.plot(predictions.concentration, predictions.survival_a, 'b--', label='Survival A')
    plt.plot(predictions.concentration, predictions.survival_b, 'r--', label='Survival B')
    plt.plot(predictions.concentration, predictions.survival_sam, 'green', label='SAM')
    plt.plot(predictions.concentration, predictions.survival_ea, 'purple', label='Effect Addition')
    plt.plot(predictions.concentration, predictions.survival_ca, 'black', label='Concentration Addition')

    plt.scatter(predictions.lc_sam_conc, predictions.lc_sam_surv, color='orange', label='LC SAM Points')

    plt.legend()
    plt.show()




def predict_file(CURVE_FITTING, TRANSFORM, generate_predictions, clean_path, file):
    data = read_data(file)
    for name, experiment in data.additional_stress.items():
        predictions = generate_predictions(experiment.name, experiment.concentration, data.main_series.survival_rate, experiment.survival_rate, transform_name=TRANSFORM)
                    
        dir = "migration/python_marco"
        os.makedirs(dir, exist_ok=True)
        path = f"{dir}/{clean_path(file)}_{experiment.name}.csv"
        predictions.to_csv(path)

if __name__ == "__main__":
    
    import glob
    import os
    import sys
    print(os.getcwd())
    sys.path.append("./")
    from data_formats import read_data, ExperimentData
    
    clean_path = lambda x: os.path.basename(x).split(".")[0]
    
    for file in glob.glob("data/*.xlsx"):
        
        try:
            predict_file("lmcurve", "williams_and_linear_interpolation", generate_predictions, clean_path, file)
        except Exception as e:
            print(f"Error in {file}: {e}")
            continue    
