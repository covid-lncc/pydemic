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
import arviz as az
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
sp_state_population = 44.04e6
rj_state_population = 16.46e6
ce_state_population = 8.843e6

target_population = brazil_population
target_population

# %%
df_brazil_states_cases = pd.read_csv(
    f"{DATA_PATH}/covid19br/cases-brazil-states.csv",
    usecols=["date", "state", "totalCases", "deaths", "recovered"],
    parse_dates=["date"],
)
df_brazil_states_cases.fillna(value={"recovered": 0}, inplace=True)
df_brazil_states_cases = df_brazil_states_cases[df_brazil_states_cases.state != "TOTAL"]


# %%
def get_brazil_state_dataframe(
    df_brazil: pd.DataFrame, state_name: str, confirmed_lower_threshold: int = 10
) -> pd.DataFrame:
    df_brazil = df_brazil.copy()
    df_state_cases = df_brazil[df_brazil.state == state_name]
    df_state_cases.reset_index(inplace=True)
    columns_rename = {"totalCases": "confirmed"}
    df_state_cases.rename(columns=columns_rename, inplace=True)
    df_state_cases["active"] = (
        df_state_cases["confirmed"] - df_state_cases["deaths"] - df_state_cases["recovered"]
    )

    df_state_cases = df_state_cases[df_state_cases.confirmed > confirmed_lower_threshold]
    day_range_list = list(range(len(df_state_cases.confirmed)))
    df_state_cases["day"] = day_range_list
    return df_state_cases


df_sp_state_cases = get_brazil_state_dataframe(df_brazil_states_cases, state_name="SP")
df_rj_state_cases = get_brazil_state_dataframe(df_brazil_states_cases, state_name="RJ")
df_ce_state_cases = get_brazil_state_dataframe(df_brazil_states_cases, state_name="CE")

# %% [markdown]
# Initial Conditions:

# %%
df_brazil_cases_by_day = pd.read_csv(f"{DATA_PATH}/brazil_by_day.csv", parse_dates=["date"])
df_brazil_cases_by_day = df_brazil_cases_by_day[df_brazil_cases_by_day.confirmed > 5]
df_brazil_cases_by_day = df_brazil_cases_by_day.reset_index(drop=True)
df_brazil_cases_by_day["day"] = df_brazil_cases_by_day.date.apply(
    lambda x: (x - df_brazil_cases_by_day.date.min()).days
)

# %%
df_rio_cases_by_day = pd.read_csv(f"{DATA_PATH}/rio_covid19.csv")
df_rio_cases_by_day["active"] = (
    df_rio_cases_by_day["cases"] - df_rio_cases_by_day["deaths"] - df_rio_cases_by_day["recoveries"]
)
rio_columns_rename = {"cases": "confirmed", "recoveries": "recovered"}
df_rio_cases_by_day.rename(columns=rio_columns_rename, inplace=True)

# %%
df_target_country = df_brazil_cases_by_day

E0, I0, R0, P0, D0, Q0, C0, H0 = (
    int(10 * float(df_target_country.confirmed.values[0])),  # E0
    int(5 * float(df_target_country.confirmed.values[0])),  # I0
    int(float(df_target_country.recovered.values[0])),  # R0
    int(float(df_target_country.active.values[0])),  # P0
    int(float(df_target_country.deaths.values[0])),  # D0
    int(float(0)),  # Q0
    int(float(df_target_country.confirmed.values[0])),  # C0
    int(float(df_target_country.recovered.values[0])),  # H0
)

S0 = target_population - (E0 + I0 + R0 + P0 + D0 + Q0)
y0_seirpdq = S0, E0, I0, R0, P0, D0, Q0, C0, H0  # SEIRPDQ IC array (not fully used)
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
    gamma_I=0.1,
    gamma_P=0.14,
    d_I=0.0105,
    d_P=0.003,
    omega=1 / 10,
    epsilon_I=1 / 3,
    eta=2e-2,
    sigma=1 / 7,
    theta=1,
    N=1,
):
    """
    SEIRPD-Q python implementation.
    """
    S, E, I, R, P, D, Q, C, H = X
    beta = beta0
    # beta = beta0  # * np.exp(-beta1 * t)
    # beta = beta0 if t < 20 else theta * beta0
    # omega = omega if 19 < t < 22 else 1e-7
    S_prime = -beta / N * S * I - omega * S + eta * R
    E_prime = beta / N * S * I - sigma * E - omega * E
    I_prime = sigma * E - gamma_I * I - d_I * I - epsilon_I * I - omega * I
    R_prime = gamma_I * I + gamma_P * P
    P_prime = epsilon_I * I - gamma_P * P - d_P * P
    D_prime = d_I * I + d_P * P
    Q_prime = omega * (S + E + I) - eta * Q
    C_prime = epsilon_I * I
    H_prime = gamma_P * P
    return S_prime, E_prime, I_prime, R_prime, P_prime, D_prime, Q_prime, C_prime, H_prime


# %% [markdown]
# ODE solvers wrapper using `scipy.integrate.solve_ivp`:

# %%
def seirpdq_ode_solver(
    y0,
    t_span,
    t_eval,
    beta0=1e-7,
    d_I=2e-4,
    d_P=9e-3,
    epsilon_I=1 / 7,
    gamma_I=1 / 14,
    omega=0,
    sigma=1 / 5,
    theta=1,
    gamma_P=1 / 15,
    eta=0,
    beta1=0,
    N=1,
):
    mu0 = beta0
    solution_ODE = solve_ivp(
        fun=lambda t, y: seirpdq_model(
            t,
            y,
            beta0=beta0,
            beta1=beta1,
            gamma_I=gamma_I,
            gamma_P=gamma_P,
            d_I=d_I,
            d_P=d_P,
            omega=omega,
            epsilon_I=epsilon_I,
            eta=eta,
            sigma=sigma,
            theta=theta,
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
            _,  # S
            _,  # E
            _,  # I
            _,  # R
            simulated_qoi1,  # P
            simulated_qoi2,  # D
            _,  # Q
            simulated_qoi3,  # C
            simulated_qoi4,  # H
        ) = simulated_ode_solution

        residual1 = f_exp1 - simulated_qoi1  # Active
        residual2 = f_exp2 - simulated_qoi2  # Deaths
        residual3 = f_exp3 - simulated_qoi3  # Cases
        residual4 = f_exp4 - simulated_qoi4  # Healed

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
    (1e-5, 0.1),  # d_I
    (1e-5, 0.1),  # d_P
    (1 / 10, 1 / 6),  # epsilon_I
    (1 / 21, 1 / 14),  # gamma_I
    (0, 1),  # omega
    (1 / 7, 1 / 4),  # sigma
    (0, 1),  # theta
    # (1 / 21, 1 / 14),  # gamma_P
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
# (
#     beta_deterministic,
#     d_I_deterministic,
#     d_P_deterministic,
#     epsilon_I_deterministic,
#     gamma_I_deterministic,
#     omega_deterministic,
#     sigma_deterministic,
#     # theta_deterministic
# ) = result_seirpdq.x

# # gamma_I_deterministic = 1 / 14
# gamma_P_deterministic = 1 / 14
# # d_I_deterministic = 2e-4
# # d_P_deterministic = 9e-3
# epsilon_I_deterministic = 1 / 7
# # sigma_deterministic = 1 / 7
# eta_deterministic = 0
# omega_deterministic = 0

# %%
def calculate_reproduction_number(
    S0, beta, mu, gamma_A, gamma_I, d_I, epsilon_I, rho, omega, sigma=1 / 7
):
    left_term = sigma * (1 - rho) * mu / ((sigma + omega) * (gamma_A + omega))
    right_term = beta * sigma * rho / ((sigma + omega) * (gamma_I + d_I + omega + epsilon_I))
    return (left_term + right_term) * S0


# reproduction_number = calculate_reproduction_number(
#     S0,
#     beta_deterministic,
#     beta_deterministic,
#     gamma_I_deterministic,
#     gamma_A_deterministic,
#     d_I_deterministic,
#     epsilon_I_deterministic,
#     rho_deterministic,
#     omega_deterministic,
#     sigma_deterministic,
# )

# %%
t0 = data_time.min()
tf = data_time.max()

solution_ODE_seirpdq = seirpdq_ode_solver(y0_seirpdq, (t0, tf), data_time, *result_seirpdq.x)
t_computed_seirpdq, y_computed_seirpdq = solution_ODE_seirpdq.t, solution_ODE_seirpdq.y
(
    S_seirpdq,
    E_seirpdq,
    I_seirpdq,
    R_seirpdq,
    P_seirpdq,
    D_seirpdq,
    Q_seirpdq,
    C_seirpdq,
    H_seirpdq,
) = y_computed_seirpdq


# %%
# parameters_dict = {
#     "Model": "SEAIRPD-Q",
#     r"$\beta$": beta_deterministic,
#     r"$\mu$": beta_deterministic,
#     r"$\gamma_I$": gamma_I_deterministic,
#     r"$\gamma_A$": 0,
#     r"$\gamma_P$": gamma_P_deterministic,
#     r"$d_I$": d_I_deterministic,
#     r"$d_P$": d_P_deterministic,
#     r"$\epsilon_I$": epsilon_I_deterministic,
#     r"$\rho$": 0,
#     r"$\omega$": omega_deterministic,
#     r"$\sigma$": sigma_deterministic,
# }

# df_parameters_calibrated = pd.DataFrame.from_records([parameters_dict])

# df_parameters_calibrated


# %%
# print(df_parameters_calibrated.to_latex(index=False))

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
    I_predict_seirpdq,
    R_predict_seirpdq,
    P_predict_seirpdq,
    D_predict_seirpdq,
    Q_predict_seirpdq,
    C_predict_seirpdq,
    H_predict_seirpdq,
) = y_computed_predict_seirpdq

# %% [markdown]
# Calculating the day when the number of infected individuals is max:

# %%
has_to_plot_infection_peak = True

crisis_day_seirpdq = np.argmax(P_predict_seirpdq)


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
plt.figure(figsize=(9, 7))

plt.plot(
    t_computed_predict_seirpdq,
    S_predict_seirpdq,
    label="Susceptible (SEAIRPD-Q)",
    marker="*",
    linestyle="-",
    markersize=10,
)

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("susceptible_projection.png")
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
# print(f"-- Reproduction number (R0):\t {reproduction_number:.3f}")
