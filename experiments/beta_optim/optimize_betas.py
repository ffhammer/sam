import glob
from stress_addition_model import (
    sam_prediction,
    SAM_Setting,
)
from data_formats import ExperimentData, read_data
from sklearn.metrics import r2_score
import optuna
import numpy as np

# def experiment(trial):


#     setting =  SAM_Setting(beta_p=trial.suggest_float("p",1,10), beta_q=trial.suggest_float("q",1,10), param_d_norm=False, stress_form= "stress_sub", stress_intercept_in_survival=0.9995, max_control_survival=0.995)


#     r2s = []

#     for path in glob.glob("data/*.xlsx"):

#         data: ExperimentData = read_data(path)

#         for name, val in data.additional_stress.items():

#             main_fit, stress_fit, sam_sur, sam_stress, additional_stress = sam_prediction(
#                 data.main_series,
#                 val,
#                 data.meta,
#                 settings=setting,
#             )

#             r2s.append(r2_score(stress_fit.survival_curve, sam_sur))

#     return np.mean(r2s) * -1

# study = optuna.create_study(
#            storage="sqlite:///db.sqlite3",
#         study_name="sam_betas"
# )
# study.optimize(experiment, n_trials=100,n_jobs= 7, show_progress_bar=True)

# print(study.best_params)


# def experiment(trial):

#     setting = SAM_Setting(
#         beta_p=3.2,
#         beta_q=3.2,
#         param_d_norm=False,
#         stress_form="stress_sub",
#         stress_intercept_in_survival=trial.suggest_float("intercept", 0.99, 1, log = True),
#         max_control_survival=trial.suggest_float("max control", 0.98, 1, log = True),
#     )

#     r2s = []

#     for path in glob.glob("data/*.xlsx"):

#         data: ExperimentData = read_data(path)

#         for name, val in data.additional_stress.items():

#             main_fit, stress_fit, sam_sur, sam_stress, additional_stress = (
#                 sam_prediction(
#                     data.main_series,
#                     val,
#                     data.meta,
#                     settings=setting,
#                 )
#             )

#             r2s.append(r2_score(stress_fit.survival_curve, sam_sur))

#     return np.mean(r2s) * -1


# study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name="intercepts")
# study.optimize(experiment, n_trials=100, n_jobs=7, show_progress_bar=True)

# print(study.best_params)


def experiment(trial):
    setting = SAM_Setting(
        beta_p=trial.suggest_float("p", 1, 10),
        beta_q=trial.suggest_float("q", 1, 10),
        param_d_norm=False,
        stress_form="stress_sub",
        stress_intercept_in_survival=trial.suggest_float(
            "intercept", 0.99, 1, log=True
        ),
        max_control_survival=trial.suggest_float("max control", 0.98, 1, log=True),
    )

    r2s = []

    for path in glob.glob("data/*.xlsx"):
        data: ExperimentData = read_data(path)

        for name, val in data.additional_stress.items():
            main_fit, stress_fit, sam_sur, sam_stress, additional_stress = (
                sam_prediction(
                    data.main_series,
                    val,
                    data.meta,
                    settings=setting,
                )
            )

            r2s.append(r2_score(stress_fit.survival_curve, sam_sur))

    return np.mean(r2s) * -1


study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name="alles")
study.optimize(experiment, n_trials=300, n_jobs=7, show_progress_bar=True)

print(study.best_params)
