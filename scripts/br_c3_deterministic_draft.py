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

# DATA_PATH = os.environ["DATA_DIR"]
DATA_PATH = "../pydemic/data/"

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
    df_brazil: pd.DataFrame, state_name: str, confirmed_lower_threshold: int = 5
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

E0, A0, I0, P0, R0, D0, C0, H0 = (
    int(10 * float(df_target_country.confirmed.values[0])),
    int(1 * float(df_target_country.confirmed.values[0])),
    int(5 * float(df_target_country.confirmed.values[0])),
    int(float(df_target_country.confirmed.values[0])),
    int(float(df_target_country.recovered.values[0])),
    int(float(df_target_country.deaths.values[0])),
    int(float(df_target_country.confirmed.values[0])),
    int(float(df_target_country.recovered.values[0])),
)

S0 = target_population - (E0 + A0 + I0 + R0 + P0 + D0)
y0_seirpdq = S0, E0, A0, I0, P0, R0, D0, C0, H0  # SEIRPDQ IC array (not fully used)
# print(y0_seirpdq)

# %% [markdown]
# <a id="implementations"></a>
# ## Programming SEIRPD-Q model in Python

# %%
@jit(nopython=True)
def omega_t(t, omega_value, decay_constant=0.000005):
    return omega_value if t < 100 else omega_value * np.exp(-decay_constant * t)


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
    omega = omega_t(t, omega)
    eta = 7e-3 if t > 200 else 0
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
    # gamma_P=1 / 14,
    d_P=9e-3,
    d_I=2e-4,
    gamma_P=1 / 14,
    mu0=1e-7,
    gamma_I=1 / 14,
    gamma_A=1 / 14,
    epsilon_I=1 / 3,
    rho=0.85,
    sigma=1 / 5,
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
# Set values from Bayesian calibration here:
beta_deterministic = 5.97e-9
mu_deterministic = beta_deterministic
omega_deterministic = 0.01976
gamma_I_deterministic = 1 / 14
gamma_A_deterministic = 1 / 14
gamma_P_deterministic = 1 / 14
d_I_deterministic = 0.013
d_P_deterministic = 0.0043
epsilon_I_deterministic = 1 / 3
rho_deterministic = 0.85
sigma_deterministic = 1 / 5
eta_deterministic = 0

parameters = [
    beta_deterministic,
    omega_deterministic,
    d_P_deterministic,
    d_I_deterministic,
    gamma_P_deterministic,
    mu_deterministic,
    gamma_I_deterministic,
    gamma_A_deterministic,
    epsilon_I_deterministic,
    rho_deterministic,
    sigma_deterministic,
    eta_deterministic
]

# beta0=1e-7,
# omega=1 / 10,
# # gamma_P=1 / 14,
# d_P=9e-3,
# d_I=2e-4,
# gamma_P=1 / 14,
# mu0=1e-7,
# gamma_I=1 / 14,
# gamma_A=1 / 14,
# epsilon_I=1 / 3,
# rho=0.85,
# sigma=1 / 5,
# eta=0,

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
data_time = df_target_country.day.values.astype(np.float64)
infected_individuals = df_target_country.active.values
dead_individuals = df_target_country.deaths.values
confirmed_cases = df_target_country.confirmed.values
recovered_cases = df_target_country.recovered.values

t0 = float(data_time.min())
number_of_days_after_last_record = 1200
tf = data_time.max() + number_of_days_after_last_record
time_range = np.linspace(t0, tf, int(tf - t0) + 1)

solution_ODE_predict_seirpdq = seirpdq_ode_solver(
    y0_seirpdq, (t0, tf), time_range, *parameters
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

Rt_predict = calculate_reproduction_number(
    S_predict_seirpdq,
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

""" plt.plot(
    data_time, infected_individuals, label="Diagnosed data", marker="s", linestyle="", markersize=10
) """
plt.plot(
    data_time, dead_individuals, label="Recorded deaths", marker="v", linestyle="", markersize=10
)

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("seirpdq_deterministic_predictions.png")
# plt.show()

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

""" plt.plot(
    data_time, recovered_cases, label="Recorded recoveries", marker="v", linestyle="", markersize=10
) """

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("seirpdq_deterministic_cumulative_predictions.png")
# plt.show()

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

# %%
plt.figure(figsize=(9, 7))

plt.plot(
    t_computed_predict_seirpdq,
    Rt_predict,
    "r",
    marker="X",
    linestyle="-",
    markersize=5,
)

plt.xlabel("Time (days)")
plt.ylabel(r"$R(t)$")
plt.grid()

plt.tight_layout()

plt.savefig("Rt_prediction.png")

# %%
# Omega evaluation
# omega_values = omega_t(t_computed_predict_seirpdq, omega_deterministic)

# plt.figure(figsize=(9, 7))

# plt.plot(
#     t_computed_predict_seirpdq,
#     omega_values,
#     "b",
#     marker="o",
#     linestyle="-",
#     markersize=5,
# )

# plt.xlabel("Time (days)")
# plt.ylabel(r"$\omega(t)$")
# plt.grid()

# plt.tight_layout()

# plt.savefig("omega_prediction.png")
print(f"{omega_t(150, omega_deterministic)}")