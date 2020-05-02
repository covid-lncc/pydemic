# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # COVID-19 SEIRPD-Q model
#
# ## Table of Contents
#
# 1. [Importing libs](#importing)
#
# 2. [Loading data](#loading)
#
# 3. [Data cleaning](#cleaning)
#
# 4. [(Very) Basic EDA](#eda)
#
# 5. [Epidemiology models](#models)
#
# 6. [Programming SEIRPD-Q model in Python](#implementations)
#
# 7. [Least-squares fitting](#least-squares)
#
# 8. [Extrapolation/Predictions](#deterministic-predictions)
#
# 9. [Forward UQ](#uq)
#
# 10. [Bayesian Calibration](#bayes-calibration)
#
# Before analyze the models, we begin having a look at the available data.
# %% [markdown]
# <a id="importing"></a>
# ## Importing libs

import os
import time

# Plotting libs
import matplotlib.pyplot as plt

# %%
from arviz.utils import Numba
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pymc3 as pm  # for uncertainty quantification and model calibration
import theano  # to control better pymc3 backend and write a wrapper
import theano.tensor as t  # for the wrapper to a custom model to pymc3
from numba import jit  # to accelerate ODE system RHS evaluations
from scipy import optimize  # to solve minimization problem from least-squares fitting
from scipy.integrate import solve_ivp  # to solve ODE system
from tqdm import tqdm, trange

seed = 12345  # for the sake of reproducibility :)
np.random.seed(seed)

plt.style.use("seaborn-talk")  # beautify the plots!

THEANO_FLAGS = "optimizer=fast_compile"  # A theano trick

Numba.enable_numba()  # speed-up arviz plots

DATA_PATH = os.environ["DATA_DIR"]

# %% [markdown]
# <a id="loading"></a>
# ## Loading data


# %%
brazil_population = float(210147125)
rio_population = float(6718903)  # gathered from IBGE 2019

target_population = rio_population
target_population

# %% [markdown]
# Initial Conditions:

# %%
df_brazil_cases_by_day = pd.read_csv(f"{DATA_PATH}/brazil_by_day.csv", parse_dates=["date"])
df_brazil_cases_by_day = df_brazil_cases_by_day[df_brazil_cases_by_day.confirmed > 5]
df_brazil_cases_by_day = df_brazil_cases_by_day.reset_index(drop=True)
df_brazil_cases_by_day["day"] = df_brazil_cases_by_day.date.apply(
    lambda x: (x - df_brazil_cases_by_day.date.min()).days
)

df_rio_cases_by_day = pd.read_csv(f"{DATA_PATH}/rio_covid19.csv")
df_rio_cases_by_day["active"] = (
    df_rio_cases_by_day["cases"] - df_rio_cases_by_day["deaths"] - df_rio_cases_by_day["recoveries"]
)
rio_columns_rename = {"cases": "confirmed", "recoveries": "recovered"}
df_rio_cases_by_day.rename(columns=rio_columns_rename, inplace=True)

df_target_country = df_rio_cases_by_day

E0, A0, I0, P0, R0, D0, C0, H0 = (
    int(10 * float(df_target_country.confirmed[0])),
    int(1 * float(df_target_country.confirmed[0])),
    int(5 * float(df_target_country.confirmed[0])),
    int(float(df_target_country.active[0])),
    int(float(df_target_country.recovered[0])),
    int(float(df_target_country.deaths[0])),
    int(float(df_target_country.confirmed[0])),
    int(float(df_target_country.recovered[0])),
)

S0 = target_population - (E0 + A0 + I0 + R0 + P0 + D0)
y0_seirpdq = S0, E0, A0, I0, P0, R0, D0, C0, H0  # SEIRPDQ IC array (not fully used)
# print(y0_seirpdq)

# %% [markdown]
# <a id="implementations"></a>
# ## Programming SEIRPD-Q model in Python

# %%
@jit(nopython=True)
def seirpdq_model(
    t,
    X,
    beta0=1e-7,
    beta1=1e-7,
    mu0=1e-7,
    mu1=1e-7,
    gamma_I=0.1,
    gamma_A=0.15,
    gamma_P=0.14,
    d_I=0.0105,
    d_P=0.003,
    omega=1 / 10,
    epsilon_I=1 / 3,
    rho=0.1,
    eta=2e-2,
    sigma=1 / 7,
    N=1,
):
    """
    SEIRPD-Q python implementation.
    """
    S, E, A, I, P, R, D, C, H = X
    beta = beta0  # * np.exp(-beta1 * t)
    mu = mu0  # * np.exp(-mu1 * t)
    S_prime = -beta / N * S * I - mu / N * S * A - omega * S + eta * R
    E_prime = beta / N * S * I + mu / N * S * A - sigma * E - omega * E
    A_prime = sigma * (1 - rho) * E - gamma_A * A - omega * A
    I_prime = sigma * rho * E - gamma_I * I - d_I * I - omega * I - epsilon_I * I
    P_prime = epsilon_I * I - gamma_P * P - d_P * P
    R_prime = gamma_A * A + gamma_I * I + gamma_P * P + omega * (S + E + A + I) - eta * R
    D_prime = d_I * I + d_P * P
    C_prime = epsilon_I * I
    H_prime = gamma_P * P
    return S_prime, E_prime, A_prime, I_prime, P_prime, R_prime, D_prime, C_prime, H_prime


# %% [markdown]
# ODE solvers wrapper using `scipy.integrate.solve_ivp`:

# %%
def seirpdq_ode_solver(
    y0,
    t_span,
    t_eval,
    beta0=1e-7,
    omega=1 / 10,
    gamma_P=1 / 14,
    mu0=1e-7,
    gamma_I=1 / 14,
    gamma_A=1 / 14,
    d_I=2e-4,
    d_P=9e-3,
    epsilon_I=1 / 7,
    rho=0.75,
    sigma=1 / 7,
    eta=0,
    beta1=0,
    mu1=0,
    N=1,
):
    mu0 = beta0
    solution_ODE = solve_ivp(
        fun=lambda t, y: seirpdq_model(
            t,
            y,
            beta0=beta0,
            beta1=beta1,
            mu0=mu0,
            mu1=mu1,
            gamma_I=gamma_I,
            gamma_A=gamma_A,
            gamma_P=gamma_P,
            d_I=d_I,
            d_P=d_P,
            omega=omega,
            epsilon_I=epsilon_I,
            rho=rho,
            eta=eta,
            sigma=sigma,
            N=N,
        ),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
    )

    return solution_ODE


# %% [markdown]
# <a id="least-squares"></a>
# ## Least-Squares fitting
#
# Now, we can know how to solve the forward problem, so we can try to fit it with a non-linear Least-Squares method for parameter estimation. Let's begin with a generic Least-Square formulation:

# %%
def seirpdq_least_squares_error_ode_y0(
    par, time_exp, f_exp, fitting_model, known_initial_conditions, total_population
):
    num_of_initial_conditions_to_fit = 3
    num_of_parameters = len(par) - num_of_initial_conditions_to_fit
    args, trial_initial_conditions = [
        par[:num_of_parameters],
        par[num_of_parameters:],
    ]
    E0, A0, I0 = trial_initial_conditions
    _, P0, R0, D0, C0, H0 = known_initial_conditions
    S0 = total_population - (E0 + A0 + I0 + R0 + P0 + D0)
    initial_conditions = [S0, E0, A0, I0, P0, R0, D0, C0, H0]

    f_exp1, f_exp2, f_exp3, f_exp4 = f_exp
    time_span = (time_exp.min(), time_exp.max())

    # weighting_denominator = f_exp1.max() + f_exp2.max() + f_exp3.max() + f_exp4.max()
    # weighting_for_exp1_constraints = 1 / (f_exp1.max() / weighting_denominator)
    # weighting_for_exp2_constraints = 1 / (f_exp2.max() / weighting_denominator)
    # weighting_for_exp3_constraints = 1 / (f_exp3.max() / weighting_denominator)
    # weighting_for_exp4_constraints = 1 / (f_exp4.max() / weighting_denominator)
    weighting_for_exp1_constraints = 0e0
    weighting_for_exp2_constraints = 1e0
    weighting_for_exp3_constraints = 1e0
    weighting_for_exp4_constraints = 0e0
    num_of_qoi = len(f_exp1)

    try:
        y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
        simulated_time = y_model.t
        simulated_ode_solution = y_model.y
        (
            _,
            _,
            _,
            _,
            simulated_qoi1,
            _,
            simulated_qoi2,
            simulated_qoi3,
            simulated_qoi4,
        ) = simulated_ode_solution

        residual1 = f_exp1 - simulated_qoi1
        residual2 = f_exp2 - simulated_qoi2  # Deaths
        residual3 = f_exp3 - simulated_qoi3  # Cases
        residual4 = f_exp4 - simulated_qoi4

        first_term = weighting_for_exp1_constraints * np.sum(residual1 ** 2.0)
        second_term = weighting_for_exp2_constraints * np.sum(residual2 ** 2.0)
        third_term = weighting_for_exp3_constraints * np.sum(residual3 ** 2.0)
        fourth_term = weighting_for_exp4_constraints * np.sum(residual4 ** 2.0)
        # first_term = weighting_for_exp1_constraints * np.abs(residual1).sum()
        # second_term = weighting_for_exp2_constraints * np.abs(residual2).sum()
        # third_term = weighting_for_exp3_constraints * np.abs(residual3).sum()
        # fourth_term = weighting_for_exp4_constraints * np.abs(residual4).sum()
        objective_function = 1 / num_of_qoi * (first_term + second_term + third_term + fourth_term)
    except ValueError:
        objective_function = 1e15

    return objective_function


def seirpdq_least_squares_error_ode(
    par, time_exp, f_exp, fitting_model, initial_conditions, total_population
):
    args = par
    f_exp1, f_exp2, f_exp3, f_exp4 = f_exp
    time_span = (time_exp.min(), time_exp.max())

    # weighting_denominator = f_exp1.max() + f_exp2.max() + f_exp3.max() + f_exp4.max()
    # weighting_for_exp1_constraints = 1 / (f_exp1.max() / weighting_denominator)
    # weighting_for_exp2_constraints = 1 / (f_exp2.max() / weighting_denominator)
    # weighting_for_exp3_constraints = 1 / (f_exp3.max() / weighting_denominator)
    # weighting_for_exp4_constraints = 1 / (f_exp4.max() / weighting_denominator)
    weighting_for_exp1_constraints = 0e0
    weighting_for_exp2_constraints = 1e0
    weighting_for_exp3_constraints = 1e0
    weighting_for_exp4_constraints = 0e0
    num_of_qoi = len(f_exp1)

    try:
        y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
        simulated_time = y_model.t
        simulated_ode_solution = y_model.y
        (
            _,
            _,
            _,
            _,
            simulated_qoi1,  # P
            _,
            simulated_qoi2,  # D
            simulated_qoi3,  # C
            simulated_qoi4,  # H
        ) = simulated_ode_solution

        residual1 = f_exp1 - simulated_qoi1
        residual2 = f_exp2 - simulated_qoi2  # Deaths
        residual3 = f_exp3 - simulated_qoi3  # Cases
        residual4 = f_exp4 - simulated_qoi4

        first_term = weighting_for_exp1_constraints * np.sum(residual1 ** 2.0)
        second_term = weighting_for_exp2_constraints * np.sum(residual2 ** 2.0)
        third_term = weighting_for_exp3_constraints * np.sum(residual3 ** 2.0)
        fourth_term = weighting_for_exp4_constraints * np.sum(residual4 ** 2.0)
        # first_term = weighting_for_exp1_constraints * np.abs(residual1).sum()
        # second_term = weighting_for_exp2_constraints * np.abs(residual2).sum()
        # third_term = weighting_for_exp3_constraints * np.abs(residual3).sum()
        # fourth_term = weighting_for_exp4_constraints * np.abs(residual4).sum()
        objective_function = 1 / num_of_qoi * (first_term + second_term + third_term + fourth_term)
    except ValueError:
        objective_function = 1e15

    return objective_function


def callback_de(xk, convergence):
    print(f"parameters = {xk}\n")


# %% [markdown]
# Setting fitting domain (given time for each observation) and the observations (observed population at given time):

# %%
data_time = df_target_country.day.values.astype(np.float64)
infected_individuals = df_target_country.active.values
dead_individuals = df_target_country.deaths.values
confirmed_cases = df_target_country.confirmed.values
recovered_cases = df_target_country.recovered.values

# %% [markdown]
# To calibrate the model, we define an objective function, which is a Least-Squares function in the present case, and minimize it. To (*try to*) avoid local minima, we use Differential Evolution (DE) method (see this [nice presentation](https://www.maths.uq.edu.au/MASCOS/Multi-Agent04/Fleetwood.pdf) to get yourself introduced to this great subject). In summary, DE is a family of Evolutionary Algorithms that aims to solve Global Optimization problems. Moreover, DE is derivative-free and population-based method.
#
# Below, calibration is performed for selected models:

# %%
# num_of_parameters_to_fit_seirpdq = 10
# bounds_seirpdq = num_of_parameters_to_fit_seirpdq * [(0, 1)]

bounds_seirpdq = [
    (0, 1e-5),  # beta
    (0, 1),  # omega
    (1 / 21, 1 / 14),  # gamma_P
]
# bounds_seirdaq = [(0, 1e-2), (0, 1), (0, 1), (0, 0.2), (0, 0.2), (0, 0.2)]

result_seirpdq = optimize.differential_evolution(
    seirpdq_least_squares_error_ode,
    bounds=bounds_seirpdq,
    args=(
        data_time,
        [infected_individuals, dead_individuals, confirmed_cases, recovered_cases],
        seirpdq_ode_solver,
        y0_seirpdq,
        target_population,
    ),
    popsize=20,
    strategy="best1bin",
    tol=5e-5,
    recombination=0.95,
    mutation=0.6,
    maxiter=10000,
    polish=True,
    disp=True,
    seed=seed,
    callback=callback_de,
    workers=16,
)

print(result_seirpdq)

# %%
print(f"-- Initial conditions: {y0_seirpdq}")

# %%
(beta_deterministic, omega_deterministic, gamma_P_deterministic,) = result_seirpdq.x

gamma_I_deterministic = 1 / 14
gamma_A_deterministic = 1 / 14
d_I_deterministic = 2e-4
d_P_deterministic = 9e-3
epsilon_I_deterministic = 1 / 7
rho_deterministic = 0.75
sigma_deterministic = 1 / 7
eta_deterministic = 0

# %%
def calculate_reproduction_number(
    S0, beta, mu, gamma_A, gamma_I, d_I, epsilon_I, rho, omega, sigma=1 / 7
):
    left_term = sigma * (1 - rho) * mu / ((sigma + omega) * (gamma_A + omega))
    right_term = beta * sigma * rho / ((sigma + omega) * (gamma_I + d_I + omega + epsilon_I))
    return (left_term + right_term) * S0


reproduction_number = calculate_reproduction_number(
    S0,
    beta_deterministic,
    beta_deterministic,
    gamma_I_deterministic,
    gamma_A_deterministic,
    d_I_deterministic,
    epsilon_I_deterministic,
    rho_deterministic,
    omega_deterministic,
    sigma_deterministic,
)

# %%
t0 = data_time.min()
tf = data_time.max()

solution_ODE_seirpdq = seirpdq_ode_solver(y0_seirpdq, (t0, tf), data_time, *result_seirpdq.x)
t_computed_seirpdq, y_computed_seirpdq = solution_ODE_seirpdq.t, solution_ODE_seirpdq.y
(
    S_seirpdq,
    E_seirpdq,
    A_seirpdq,
    I_seirpdq,
    P_seirpdq,
    R_seirpdq,
    D_seirpdq,
    C_seirpdq,
    H_seirpdq,
) = y_computed_seirpdq


# %%
parameters_dict = {
    "Model": "SEAIRPD-Q",
    r"$\beta$": beta_deterministic,
    r"$\mu$": beta_deterministic,
    r"$\gamma_I$": gamma_I_deterministic,
    r"$\gamma_A$": gamma_A_deterministic,
    r"$\gamma_P$": gamma_P_deterministic,
    r"$d_I$": d_I_deterministic,
    r"$d_P$": d_P_deterministic,
    r"$\epsilon_I$": epsilon_I_deterministic,
    r"$\rho$": rho_deterministic,
    r"$\omega$": omega_deterministic,
    r"$\sigma$": sigma_deterministic,
}

df_parameters_calibrated = pd.DataFrame.from_records([parameters_dict])

# df_parameters_calibrated


# %%
print(df_parameters_calibrated.to_latex(index=False))

# %% [markdown]
# Show calibration result based on available data:

# %%
plt.figure(figsize=(9, 7))

plt.plot(
    t_computed_seirpdq,
    I_seirpdq,
    label="Infected (SEAIRPD-Q)",
    marker="X",
    linestyle="-",
    markersize=10,
)
plt.plot(
    t_computed_seirpdq,
    A_seirpdq,
    label="Asymptomatic (SEAIRPD-Q)",
    marker="o",
    linestyle="-",
    markersize=10,
)
# plt.plot(t_computed_seirdq, R_seirdq * target_population, label='Recovered (SEIRDAQ)', marker='o', linestyle="-", markersize=10)
plt.plot(
    t_computed_seirpdq,
    P_seirpdq,
    label="Diagnosed (SEAIRPD-Q)",
    marker="s",
    linestyle="-",
    markersize=10,
)
plt.plot(
    t_computed_seirpdq,
    D_seirpdq,
    label="Deaths (SEAIRPD-Q)",
    marker="s",
    linestyle="-",
    markersize=10,
)

plt.plot(
    data_time, infected_individuals, label="Diagnosed data", marker="s", linestyle="", markersize=10
)
plt.plot(
    data_time, dead_individuals, label="Recorded deaths", marker="v", linestyle="", markersize=10
)
plt.legend()
plt.grid()
plt.xlabel("Time (days)")
plt.ylabel("Population")

plt.tight_layout()
plt.savefig("seirpdq_deterministic_calibration.png")
plt.show()

# %%
plt.figure(figsize=(9, 7))

plt.plot(
    t_computed_seirpdq,
    C_seirpdq,
    label="Cases (SEAIRPD-Q)",
    marker="X",
    linestyle="-",
    markersize=10,
)

plt.plot(
    t_computed_seirpdq,
    H_seirpdq,
    label="Recovered (SEAIRPD-Q)",
    marker="D",
    linestyle="-",
    markersize=10,
)

plt.plot(
    t_computed_seirpdq,
    D_seirpdq,
    label="Deaths (SEAIRPD-Q)",
    marker="v",
    linestyle="-",
    markersize=10,
)

plt.plot(
    data_time, confirmed_cases, label="Confirmed data", marker="s", linestyle="", markersize=10
)

plt.plot(
    data_time, dead_individuals, label="Recorded deaths", marker="v", linestyle="", markersize=10
)

plt.plot(
    data_time, recovered_cases, label="Recorded recoveries", marker="v", linestyle="", markersize=10
)

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("seirpdq_deterministic_cumulative_calibration.png")
plt.show()

# %%
methods_list = list()
deaths_list = list()

methods_list.append("SEIRPD-Q")
deaths_list.append(int(D_seirpdq.max()))
print(f"-- Confirmed cases estimate for today (SEIRPD-Q):\t{int(P_seirpdq.max())}")
print(
    f"-- Confirmed cases estimate population percentage for today (SEIRPD-Q):\t{100 * P_seirpdq.max() / target_population:.3f}%"
)
print(f"-- Death estimate for today (SEIRPD-Q):\t{int(D_seirpdq.max())}")
print(
    f"-- Death estimate population percentage for today (SEIRPD-Q):\t{100 * D_seirpdq.max() / target_population:.3f}%"
)

methods_list.append("Recorded")
deaths_list.append(int(dead_individuals[-1]))

death_estimates_dict = {"Method": methods_list, "Deaths estimate": deaths_list}
df_deaths_estimates = pd.DataFrame(death_estimates_dict)
print(f"-- Recorded deaths until today:\t{int(dead_individuals[-1])}")


# %%
# df_deaths_estimates.set_index("Model", inplace=True)
print(df_deaths_estimates.to_latex(index=False))

# %% [markdown]
# <a id="deterministic-predictions"></a>
# ## Extrapolation/Predictions
#
# Now, let's extrapolate to next days.

# %%
t0 = float(data_time.min())
number_of_days_after_last_record = 120
tf = data_time.max() + number_of_days_after_last_record
time_range = np.linspace(t0, tf, int(tf - t0) + 1)

solution_ODE_predict_seirpdq = seirpdq_ode_solver(
    y0_seirpdq, (t0, tf), time_range, *result_seirpdq.x
)  # SEIRDAQ
#     solution_ODE_predict_seirdaq = seirdaq_ode_solver(y0_seirdaq, (t0, tf), time_range)  # SEIRDAQ
t_computed_predict_seirpdq, y_computed_predict_seirpdq = (
    solution_ODE_predict_seirpdq.t,
    solution_ODE_predict_seirpdq.y,
)
(
    S_predict_seirpdq,
    E_predict_seirpdq,
    A_predict_seirpdq,
    I_predict_seirpdq,
    P_predict_seirpdq,
    R_predict_seirpdq,
    D_predict_seirpdq,
    C_predict_seirpdq,
    H_predict_seirpdq,
) = y_computed_predict_seirpdq

# %% [markdown]
# Calculating the day when the number of infected individuals is max:

# %%
has_to_plot_infection_peak = True

crisis_day_seirpdq = np.argmax(P_predict_seirpdq) + 1


# %%
plt.figure(figsize=(9, 7))

#     plt.plot(t_computed_predict_seirdaq, 100 * S_predict_seirdq, label='Susceptible (SEIRD-Q)', marker='s', linestyle="-", markersize=10)
# plt.plot(t_computed_predict_seirpdq, E_predict_seirpdq, label='Exposed (SEIRPD-Q)', marker='*', linestyle="-", markersize=10)
plt.plot(
    t_computed_predict_seirpdq,
    I_predict_seirpdq,
    label="Infected (SEAIRPD-Q)",
    marker="X",
    linestyle="-",
    markersize=10,
)
plt.plot(
    t_computed_predict_seirpdq,
    A_predict_seirpdq,
    label="Asymptomatic (SEAIRPD-Q)",
    marker="o",
    linestyle="-",
    markersize=10,
)
#     plt.plot(t_computed_predict_seirdaq, 100 * R_predict_seirdaq, label='Recovered (SEIRDAQ)', marker='o', linestyle="-", markersize=10)
plt.plot(
    t_computed_predict_seirpdq,
    D_predict_seirpdq,
    label="Deaths (SEAIRPD-Q)",
    marker="v",
    linestyle="-",
    markersize=10,
)
plt.plot(
    t_computed_predict_seirpdq,
    P_predict_seirpdq,
    label="Diagnosed (SEAIRPD-Q)",
    marker="D",
    linestyle="-",
    markersize=10,
)
if has_to_plot_infection_peak:
    plt.axvline(
        x=crisis_day_seirpdq, color="red", linestyle="-", label="Diagnosed peak (SEAIRPD-Q)"
    )

plt.plot(
    data_time, infected_individuals, label="Diagnosed data", marker="s", linestyle="", markersize=10
)
plt.plot(
    data_time, dead_individuals, label="Recorded deaths", marker="v", linestyle="", markersize=10
)

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("seirpdq_deterministic_predictions.png")
plt.show()

# %%
plt.figure(figsize=(9, 7))

plt.plot(
    t_computed_predict_seirpdq,
    C_predict_seirpdq,
    label="Cases (SEAIRPD-Q)",
    marker="X",
    linestyle="-",
    markersize=10,
)

plt.plot(
    t_computed_predict_seirpdq,
    H_predict_seirpdq,
    label="Recovered (SEAIRPD-Q)",
    marker="D",
    linestyle="-",
    markersize=10,
)

plt.plot(
    t_computed_predict_seirpdq,
    D_predict_seirpdq,
    label="Deaths (SEAIRPD-Q)",
    marker="v",
    linestyle="-",
    markersize=10,
)

plt.plot(
    data_time, confirmed_cases, label="Confirmed data", marker="s", linestyle="", markersize=10
)

plt.plot(
    data_time, dead_individuals, label="Recorded deaths", marker="v", linestyle="", markersize=10
)

plt.plot(
    data_time, recovered_cases, label="Recorded recoveries", marker="v", linestyle="", markersize=10
)

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("seirpdq_deterministic_cumulative_predictions.png")
plt.show()

# %%
print(
    f"-- Max number of diagnosed individuals (SEIRPD-Q model):\t {int(np.max(P_predict_seirpdq))}"
)
print(
    f"-- Population percentage of max number of diagnosed individuals (SEIRPD-Q model):\t {100 * np.max(P_predict_seirpdq) / target_population:.2f}%"
)
print(
    f"-- Day estimate for max number of diagnosed individuals (SEIRPD-Q model):\t {crisis_day_seirpdq}"
)
print(
    f"-- Percentage of number of death estimate (SEIRPD-Q model):\t {100 * D_predict_seirpdq[-1] / target_population:.3f}%"
)
print(f"-- Number of death estimate (SEIRPD-Q model):\t {int(D_predict_seirpdq[-1])}")
print(f"-- Reproduction number (R0):\t {reproduction_number:.3f}")


# %% [markdown]
# <a id="bayes-calibration"></a>
# ## Bayesian Calibration

# %%
observations_to_fit = np.vstack([dead_individuals, confirmed_cases])


# %%
@theano.compile.ops.as_op(
    itypes=[t.dvector, t.dvector, t.dscalar, t.dscalar, t.dscalar, t.dscalar], otypes=[t.dmatrix]
)
def seirpdq_ode_wrapper(time_exp, initial_conditions, beta, gamma, delta, theta):
    time_span = (time_exp.min(), time_exp.max())

    args = [beta, gamma, delta, theta]
    y_model = seirpdq_ode_solver(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    (_, _, _, _, _, _, simulated_qoi1, simulated_qoi2, _,) = simulated_ode_solution

    concatenate_simulated_qoi = np.vstack([simulated_qoi1, simulated_qoi2])

    return concatenate_simulated_qoi


@theano.compile.ops.as_op(
    itypes=[
        t.dvector,
        t.dvector,
        t.dscalar,
        t.dscalar,  # beta
        t.dscalar,  # omega
        t.dscalar,  # gamma_P
    ],
    otypes=[t.dmatrix],
)
def seirpdq_ode_wrapper_with_y0(
    time_exp, initial_conditions, total_population, beta, omega, gamma_P,
):
    time_span = (time_exp.min(), time_exp.max())

    args = [beta, omega, gamma_P]
    y_model = seirpdq_ode_solver(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    (_, _, _, _, _, _, simulated_qoi1, simulated_qoi2, _,) = simulated_ode_solution

    concatenate_simulated_qoi = np.vstack([simulated_qoi1, simulated_qoi2])

    return concatenate_simulated_qoi


# %%
print("\n*** Performing Bayesian calibration ***")

print("-- Running Monte Carlo simulations:")

start_time = time.time()
with pm.Model() as model_mcmc:
    # Prior distributions for the model's parameters
    beta = pm.Uniform("beta", lower=0, upper=1e-5,)
    omega = pm.Uniform("omega", lower=0, upper=1,)
    gamma_P = pm.Uniform("gamma_P", lower=1 / 21, upper=1 / 14,)

    standard_deviation = pm.Uniform("std_deviation", lower=1e1, upper=1e2)

    # Defining the deterministic formulation of the problem
    fitting_model = pm.Deterministic(
        "seirpdq_model",
        seirpdq_ode_wrapper_with_y0(
            theano.shared(data_time),
            theano.shared(np.array(y0_seirpdq)),
            theano.shared(target_population),
            beta,
            omega,
            gamma_P,
        ),
    )

    likelihood_model = pm.Normal(
        "likelihood_model", mu=fitting_model, sigma=standard_deviation, observed=observations_to_fit
    )

    # The Monte Carlo procedure driver
    step = pm.step_methods.DEMetropolis()
    seirdpq_trace_calibration = pm.sample(
        30000, chains=8, cores=8, step=step, random_seed=seed, tune=10000
    )
    # seirdpq_trace_calibration = pm.sample(
    #     50000, step=step, random_seed=seed, tune=35000
    # )
    # seirdpq_trace_calibration = pm.sample(
    #     4000, random_seed=seed, tune=1000
    # )

duration = time.time() - start_time

print(f"-- Monte Carlo simulations done in {duration / 60:.3f} minutes")

# %%
print("-- Arviz post-processing:")
import warnings

warnings.filterwarnings("ignore")

start_time = time.time()
plot_step = 10


# %%
calibration_variable_names = [
    "std_deviation",
    "beta",
    "gamma_P",
    "omega",
]

progress_bar = tqdm(calibration_variable_names)
for variable in progress_bar:
    progress_bar.set_description("Arviz post-processing")
    pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=(f"{variable}"))
    plt.savefig(f"seirpdq_{variable}_traceplot_cal.png")

    pm.plot_posterior(
        seirdpq_trace_calibration[::plot_step], var_names=(f"{variable}"), kind="hist", round_to=5
    )
    plt.savefig(f"seirpdq_{variable}_posterior_cal.png")


# %%
percentile_cut = 2.5

y_min = np.percentile(seirdpq_trace_calibration["seirpdq_model"], percentile_cut, axis=0)
y_max = np.percentile(seirdpq_trace_calibration["seirpdq_model"], 100 - percentile_cut, axis=0)
y_fit = np.percentile(seirdpq_trace_calibration["seirpdq_model"], 50, axis=0)


# %%
std_deviation = seirdpq_trace_calibration.get_values("std_deviation")
sd_pop = np.sqrt(std_deviation.mean())
print(f"-- Estimated standard deviation mean: {sd_pop}")


# %%
plt.figure(figsize=(9, 7))

plt.plot(
    data_time, y_fit[0], "r", label="Deaths (SEAIRPD-Q)", marker="D", linestyle="-", markersize=10,
)
plt.fill_between(data_time, y_min[0], y_max[0], color="r", alpha=0.2)

plt.plot(
    data_time, y_fit[1], "b", label="Cases (SEAIRPD-Q)", marker="v", linestyle="-", markersize=10
)
plt.fill_between(data_time, y_min[1], y_max[1], color="b", alpha=0.2)

# plt.errorbar(data_time, infected_individuals, yerr=sd_pop, label='Recorded diagnosed', linestyle='None', marker='s', markersize=10)
# plt.errorbar(data_time, dead_individuals, yerr=sd_pop, label='Recorded deaths', marker='v', linestyle="None", markersize=10)
plt.plot(
    data_time, confirmed_cases, label="Confirmed data", marker="s", linestyle="", markersize=10
)
plt.plot(
    data_time, dead_individuals, label="Recorded deaths", marker="v", linestyle="", markersize=10
)

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig("seirpdq_calibration_bayes.png")
# plt.show()

# %%
duration = time.time() - start_time

print(f"-- Arviz post-processing done in {duration / 60:.3f} minutes")
# %% [markdown]
# Now we evaluate prediction. We have to retrieve parameter realizations.

# %%
print("\n*** Performing Bayesian prediction ***")
print("-- Exporting calibrated parameter to CSV")

start_time = time.time()

dict_realizations = dict()
progress_bar = tqdm(calibration_variable_names)
for variable in progress_bar:
    progress_bar.set_description(f"Gathering {variable} realizations")
    parameter_realization = seirdpq_trace_calibration.get_values(f"{variable}")
    dict_realizations[f"{variable}"] = parameter_realization

df_realizations = pd.DataFrame(dict_realizations)
df_realizations.to_csv("calibration_realizations.csv")

duration = time.time() - start_time

print(f"-- Exported done in {duration:.3f} seconds")

print("-- Processing Bayesian predictions")

S_predicted = list()
E_predicted = list()
A_predicted = list()
I_predicted = list()
P_predicted = list()
R_predicted = list()
D_predicted = list()
C_predicted = list()
H_predicted = list()
number_of_total_realizations = len(dict_realizations["beta"])
for realization in trange(number_of_total_realizations):
    parameters_realization = [
        dict_realizations["beta"][realization],
        dict_realizations["omega"][realization],
        dict_realizations["gamma_P"][realization],
    ]
    solution_ODE_predict = seirpdq_ode_solver(
        y0_seirpdq, (t0, tf), time_range, *parameters_realization
    )
    t_computed_predict, y_computed_predict = solution_ODE_predict.t, solution_ODE_predict.y
    S, E, A, I, P, R, D, C, H = y_computed_predict

    S_predicted.append(S)
    E_predicted.append(E)
    A_predicted.append(A)
    I_predicted.append(I)
    P_predicted.append(P)
    R_predicted.append(R)
    D_predicted.append(D)
    C_predicted.append(C)
    H_predicted.append(H)

S_predicted = np.array(S_predicted)
E_predicted = np.array(E_predicted)
A_predicted = np.array(A_predicted)
I_predicted = np.array(I_predicted)
P_predicted = np.array(P_predicted)
R_predicted = np.array(R_predicted)
D_predicted = np.array(D_predicted)
C_predicted = np.array(C_predicted)
H_predicted = np.array(H_predicted)

percentile_cut = 2.5
C_min = np.percentile(C_predicted, percentile_cut, axis=0)
C_max = np.percentile(C_predicted, 100 - percentile_cut, axis=0)
C_mean = np.percentile(C_predicted, 50, axis=0)

D_min = np.percentile(D_predicted, percentile_cut, axis=0)
D_max = np.percentile(D_predicted, 100 - percentile_cut, axis=0)
D_mean = np.percentile(D_predicted, 50, axis=0)

# %%
plt.figure(figsize=(9, 7))

plt.plot(
    t_computed_predict,
    C_mean,
    "b",
    label="Cases (SEAIRPD-Q)",
    marker="D",
    linestyle="-",
    markersize=10,
)
plt.fill_between(t_computed_predict, C_min, C_max, color="b", alpha=0.2)

plt.plot(
    t_computed_predict,
    D_mean,
    "r",
    label="Deaths (SEAIRPD-Q)",
    marker="v",
    linestyle="-",
    markersize=10,
)
plt.fill_between(t_computed_predict, D_min, D_max, color="r", alpha=0.2)

# plt.errorbar(data_time, infected_individuals, yerr=sd_pop, label='Recorded diagnosed', linestyle='None', marker='s', markersize=10)
# plt.errorbar(data_time, dead_individuals, yerr=sd_pop, label='Recorded deaths', marker='v', linestyle="None", markersize=10)
plt.plot(
    data_time, confirmed_cases, label="Confirmed data", marker="s", linestyle="", markersize=10
)
plt.plot(
    data_time, dead_individuals, label="Recorded deaths", marker="v", linestyle="", markersize=10
)

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig("seirpdq_prediction_bayes.png")
plt.show()
