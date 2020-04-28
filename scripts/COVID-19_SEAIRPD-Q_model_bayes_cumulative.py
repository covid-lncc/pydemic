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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pymc3 as pm  # for uncertainty quantification and model calibration
import theano  # to control better pymc3 backend and write a wrapper
import theano.tensor as t  # for the wrapper to a custom model to pymc3
from numba import jit  # to accelerate ODE system RHS evaluations
from scipy import optimize  # to solve minimization problem from least-squares fitting
from scipy.integrate import solve_ivp  # to solve ODE system
from tqdm import trange

seed = 12345  # for the sake of reproducibility :)
np.random.seed(seed)

plt.style.use("seaborn-talk")  # beautify the plots!

THEANO_FLAGS = "optimizer=fast_compile"  # A theano trick

DATA_PATH = os.environ["DATA_DIR"]

# %% [markdown]
# <a id="loading"></a>
# ## Loading data

# %%
df_covid = pd.read_csv(f"{DATA_PATH}/covid_19_clean_complete.csv", parse_dates=["Date"])

df_covid.info()


# %%
df_covid.head()


# %%
columns_to_filter_cases = ["Country/Region", "Date", "Confirmed", "Deaths"]
df_covid_cases = df_covid[columns_to_filter_cases]

df_covid_cases.head()

# %% [markdown]
# <a id="cleaning"></a>
# ## Data cleaning
#
# Let's do a data cleaning based in this [amazing notebook!](https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper)

# %%
print(f"First day entry:\t {df_covid['Date'].min()}")
print(f"Last day reported:\t {df_covid['Date'].max()}")
print(f"Total of tracked days:\t {df_covid['Date'].max() - df_covid['Date'].min()}")


# %%
df_covid.rename(
    columns={
        "Date": "date",
        "Province/State": "state",
        "Country/Region": "country",
        "Last Update": "last_updated",
        "Confirmed": "confirmed",
        "Deaths": "deaths",
        "Recovered": "recovered",
    },
    inplace=True,
)

df_covid

# %% [markdown]
# Active Case = confirmed - deaths - recovered

# %%
df_covid["active"] = df_covid["confirmed"] - df_covid["deaths"] - df_covid["recovered"]

df_covid

# %% [markdown]
# Replacing Mainland china with just China:

# %%
df_covid["country"] = df_covid["country"].replace("Mainland China", "China")

df_covid

# %% [markdown]
# <a id="eda"></a>
# ## (Very) Basic EDA
# %% [markdown]
# Worldwide scenario:

# %%
df_grouped = df_covid.groupby("date")["date", "confirmed", "active", "deaths"].sum().reset_index()

df_grouped

# %% [markdown]
# Now, let's take a look at Brazil:

# %%
def get_df_country_cases(df: pd.DataFrame, country_name: str) -> pd.DataFrame:
    df_grouped_country = df[df["country"] == country_name].reset_index()
    df_grouped_country_date = (
        df_grouped_country.groupby("date")["date", "confirmed", "deaths", "recovered"]
        .sum()
        .reset_index()
    )
    df_grouped_country_date["active"] = (
        df_grouped_country_date["confirmed"]
        - df_grouped_country_date["deaths"]
        - df_grouped_country_date["recovered"]
    )
    df_grouped_country_date["confirmed_marker"] = df_grouped_country_date.shape[0] * ["Active"]
    df_grouped_country_date["deaths_marker"] = df_grouped_country_date.shape[0] * ["Deaths"]
    return df_grouped_country_date


def get_df_state_cases(df: pd.DataFrame, state_name: str) -> pd.DataFrame:
    df_grouped_state = df[df["state"] == state_name].reset_index()
    df_grouped_state_date = (
        df_grouped_state.groupby("date")["date", "confirmed", "deaths", "recovered"]
        .sum()
        .reset_index()
    )
    df_grouped_state_date["active"] = (
        df_grouped_state_date["confirmed"]
        - df_grouped_state_date["deaths"]
        - df_grouped_state_date["recovered"]
    )
    df_grouped_state_date["confirmed_marker"] = df_grouped_state_date.shape[0] * ["Active"]
    df_grouped_state_date["deaths_marker"] = df_grouped_state_date.shape[0] * ["Deaths"]
    return df_grouped_state_date


# %%
df_grouped_brazil = get_df_country_cases(df_covid, "Brazil")

df_grouped_brazil

# %% [markdown]
# Let's have a look at China:

# %%
df_grouped_china = get_df_country_cases(df_covid, "China")

# %% [markdown]
# Now, let's take a look only at Hubei, which is disease focus in China:

# %%
df_grouped_hubei = get_df_state_cases(df_covid, "Hubei")

# %% [markdown]
# Now a look at Italy:

# %%
df_grouped_italy = get_df_country_cases(df_covid, "Italy")

# %% [markdown]
# ### Comparison between Brazil and Italy

# %%
df_brazil_cases_by_day = df_grouped_brazil[df_grouped_brazil.confirmed > 5]
df_brazil_cases_by_day = df_brazil_cases_by_day.reset_index(drop=True)
df_brazil_cases_by_day["day"] = df_brazil_cases_by_day.date.apply(
    lambda x: (x - df_brazil_cases_by_day.date.min()).days
)

reordered_columns = [
    "date",
    "day",
    "confirmed",
    "deaths",
    "recovered",
    "active",
    "confirmed_marker",
    "deaths_marker",
]
df_brazil_cases_by_day = df_brazil_cases_by_day[reordered_columns]

# df_brazil_cases_by_day.to_csv(
#     "brazil_by_day.csv", columns=["date", "day", "confirmed", "deaths", "recovered", "active"]
# )


# %%
df_italy_cases_by_day = df_grouped_italy[df_grouped_italy.confirmed > 0]
df_italy_cases_by_day = df_italy_cases_by_day.reset_index(drop=True)
df_italy_cases_by_day["day"] = df_italy_cases_by_day.date.apply(
    lambda x: (x - df_italy_cases_by_day.date.min()).days
)

reordered_columns = [
    "date",
    "day",
    "confirmed",
    "deaths",
    "recovered",
    "active",
    "confirmed_marker",
    "deaths_marker",
]
df_italy_cases_by_day = df_italy_cases_by_day[reordered_columns]

# df_italy_cases_by_day.to_csv(
#     "italy_by_day.csv", columns=["date", "day", "confirmed", "deaths", "recovered", "active"]
# )


# %%
df_italy_cases_by_day_limited_by_br = df_italy_cases_by_day[
    df_italy_cases_by_day.day <= df_brazil_cases_by_day.day.max()
]
days = df_brazil_cases_by_day.day

plt.figure(figsize=(9, 6))
plt.plot(
    days,
    df_brazil_cases_by_day.confirmed,
    marker="X",
    linestyle="",
    markersize=10,
    label="Confirmed (BR)",
)
plt.plot(
    days,
    df_italy_cases_by_day_limited_by_br.confirmed,
    marker="o",
    linestyle="",
    markersize=10,
    label="Confirmed (ITA)",
)
plt.plot(
    days,
    df_brazil_cases_by_day.deaths,
    marker="s",
    linestyle="",
    markersize=10,
    label="Deaths (BR)",
)
plt.plot(
    days,
    df_italy_cases_by_day_limited_by_br.deaths,
    marker="*",
    linestyle="",
    markersize=10,
    label="Deaths (ITA)",
)
plt.plot(
    days,
    df_brazil_cases_by_day.active,
    marker="P",
    linestyle="",
    markersize=10,
    label="Active (BR)",
)
plt.plot(
    days,
    df_italy_cases_by_day_limited_by_br.active,
    marker="v",
    linestyle="",
    markersize=10,
    label="Active (ITA)",
)

plt.xlabel("Day(s)")
plt.ylabel("Cases")
plt.legend()
plt.grid()

plt.show()

# %% [markdown]
# ### China scenario since first entry

# %%
df_china_cases_by_day = df_grouped_china[df_grouped_china.confirmed > 0]
df_china_cases_by_day = df_china_cases_by_day.reset_index(drop=True)
df_china_cases_by_day["day"] = df_china_cases_by_day.date.apply(
    lambda x: (x - df_china_cases_by_day.date.min()).days
)

reordered_columns = [
    "date",
    "day",
    "confirmed",
    "deaths",
    "recovered",
    "active",
    "confirmed_marker",
    "deaths_marker",
]
df_china_cases_by_day = df_china_cases_by_day[reordered_columns]

# df_china_cases_by_day.to_csv(
#     "china_by_day.csv", columns=["date", "day", "confirmed", "deaths", "recovered", "active"]
# )

# %% [markdown]
# And for Hubei:

# %%
df_hubei_cases_by_day = df_grouped_hubei[df_grouped_hubei.confirmed > 0]
df_hubei_cases_by_day = df_hubei_cases_by_day.reset_index(drop=True)
df_hubei_cases_by_day["day"] = df_hubei_cases_by_day.date.apply(
    lambda x: (x - df_hubei_cases_by_day.date.min()).days
)

reordered_columns = [
    "date",
    "day",
    "confirmed",
    "deaths",
    "recovered",
    "active",
    "confirmed_marker",
    "deaths_marker",
]
df_hubei_cases_by_day = df_hubei_cases_by_day[reordered_columns]

# df_hubei_cases_by_day.to_csv(
#     "hubei_by_day.csv", columns=["date", "day", "confirmed", "deaths", "recovered", "active"]
# )

# %% [markdown]
# ### Spain since first recorded case

# %%
df_grouped_spain = get_df_country_cases(df_covid, "Spain")
df_spain_cases_by_day = df_grouped_spain[df_grouped_spain.confirmed > 0]
df_spain_cases_by_day = df_spain_cases_by_day.reset_index(drop=True)
df_spain_cases_by_day["day"] = df_spain_cases_by_day.date.apply(
    lambda x: (x - df_spain_cases_by_day.date.min()).days
)

reordered_columns = [
    "date",
    "day",
    "confirmed",
    "deaths",
    "recovered",
    "active",
    "confirmed_marker",
    "deaths_marker",
]
df_spain_cases_by_day = df_spain_cases_by_day[reordered_columns]

# %% [markdown]
# ### Iran since first case

# %%
df_grouped_iran = get_df_country_cases(df_covid, "Iran")
df_iran_cases_by_day = df_grouped_iran[df_grouped_iran.confirmed > 0]
df_iran_cases_by_day = df_iran_cases_by_day.reset_index(drop=True)
df_iran_cases_by_day["day"] = df_iran_cases_by_day.date.apply(
    lambda x: (x - df_iran_cases_by_day.date.min()).days
)

reordered_columns = [
    "date",
    "day",
    "confirmed",
    "deaths",
    "recovered",
    "active",
    "confirmed_marker",
    "deaths_marker",
]
df_iran_cases_by_day = df_iran_cases_by_day[reordered_columns]

# %% [markdown]
# ### USA since first case

# %%
df_grouped_usa = get_df_country_cases(df_covid, "US")
df_usa_cases_by_day = df_grouped_usa[df_grouped_usa.confirmed > 0]
df_usa_cases_by_day = df_usa_cases_by_day.reset_index(drop=True)
df_usa_cases_by_day["day"] = df_usa_cases_by_day.date.apply(
    lambda x: (x - df_usa_cases_by_day.date.min()).days
)

reordered_columns = [
    "date",
    "day",
    "confirmed",
    "deaths",
    "recovered",
    "active",
    "confirmed_marker",
    "deaths_marker",
]
df_usa_cases_by_day = df_usa_cases_by_day[reordered_columns]

# %% [markdown]
# <a id="models"></a>
# ## Epidemiology models
#
# Now, let me explore the data in order to calibrate an epidemiologic model in order to try to simulate and predict cases.
#
# ### Classical models
#
# Here I present a brief review of classical temporal models (space dependency is not considered). Then I proposed modifications for such models.
#
# #### SIR model
#
# The model represents an epidemic scenario, aiming to predict and control infectious diseases. It consists in a non-linear dynamical system, which considers populational sub-groups according to the state of the individuals. A simple model would be composed by 3 subgroups:
#
# * Susceptible individuals (S);
# * Infected (I);
# * Recovered (R).
#
# With such components, a classical dynamical system known as SIR model. The equations of such a system is written as:
#
# \begin{align*}
#   \dot{S} &= - \beta S I \\
#   \dot{I} &= \beta S I - \zeta I \\
#   \dot{R} &= \zeta I
# \end{align*}
#
# where $\dot{(\bullet)}$ stands for time-derivative.
#
# Some biological explanation for parameters:
#
# * $\beta$ is the conversion parameter due to interaction between a susceptible individual with an infected one;
# * $\zeta$ is the conversion parameter related to the recovery rate. In other words, the individuals that become immune;
#
# #### SEIR model
#
# Another classical model known as SEIR (Susceptible-Exposed-Infected-Recovered) is common applied in Computational Epidemiology literature (you can check it elsewhere). In this model, a new sub-group of individuals is considered: Exposed. Such individuals are those that are infected, but don't show any sympton. In the classical SEIR model, exposed individuals **do not transmit the disease**. The ODE system now becomes:
#
# \begin{align*}
#     \dot{S} &= - \beta S  I \\
#     \dot{E} &= \beta S I - \alpha E \\
#     \dot{I} &= \alpha E - \zeta I \\
#     \dot{R} &= \zeta I \\
# \end{align*}
#
# Brief biological interpretation for additional parameter:
#
# * $\alpha$ is the conversion parameter for exposed individuals that transformed into infected ones.
#
# ### Modified models
#
# Here, I propose some simple modifications in order to improve model representability for COVID-19.
#
# #### Modified SIR model (SIRD)
#
# In this model, deaths due to the disease is considered explicitly. A new individuals sub-group is introduced: dead individuals. To consider such phenomenon, an additional equation is required, as well as a modification in the Infected equation balance. The ODE system is given below:
#
# \begin{align*}
#   \dot{S} &= - \beta S I \\
#   \dot{I} &= \beta S I - \zeta I - \delta I \\
#   \dot{R} &= \zeta I \\
#   \dot{D} &= \delta I
# \end{align*}
#
# Brief biological interpretation for additional parameter:
#
# * $\delta$ is the mortality rate for the disease.
#
# #### Modified SEIR model (SEIR-2)
#
# This model aims to solve the lack of the original SEIR model, which does not consider disease transmission between exposed and susceptible individuals. In order to take it into account,
# we modified balance equations for S and E as follows:
#
# \begin{align*}
#     \dot{S} &= - \beta S  I  - \gamma S E \\
#     \dot{E} &= \beta S I - \alpha E + \gamma S E \\
#     \dot{I} &= \alpha E - \zeta I \\
#     \dot{R} &= \zeta I \\
# \end{align*}
#
# Brief biological interpretation for additional parameter:
#
# * $\gamma$ is the conversion rate parameter for susceptible individuals that interact with exposed individuals and then become exposed.
#
# #### Modified SEIR model with deaths (SEIRD)
#
# Very similiar to the last one, but it considers a sub-population of dead individuals due to the disease. Thus, the model is written as:
#
# \begin{align*}
#     \dot{S} &= - \beta S  I  - \gamma S E \\
#     \dot{E} &= \beta S I - \alpha E + \gamma S E \\
#     \dot{I} &= \alpha E - \zeta I - \delta I \\
#     \dot{R} &= \zeta I \\
#     \dot{D} &= \delta I
# \end{align*}
#
# #### Modified SEIRD model considering quarantine lockdown (SEIRD-Q)
#
# This is a modified model that take into account a removal rate from Susceptible, Exposed and Infected individuals to quarantine. The main hypothesis is that this conversion
# is under a constant removal parameter (by time, i.e., 1 / time), and after the conversion, the individual becomes "Recovered" and can not transmit the disease anymore. The new system is written as
#
# \begin{align*}
#     \dot{S} &= - \beta S I  - \gamma S E - \omega S\\
#     \dot{E} &= \beta S I - \alpha E + \gamma S E - \omega E \\
#     \dot{I} &= \alpha E - \zeta I - \delta I - \omega I \\
#     \dot{R} &= \zeta I + \omega (S + E + I) \\
#     \dot{D} &= \delta I
# \end{align*}
#
# Brief biological interpretation for additional parameter:
#
# * $\omega$ is the conversion rate parameter for Susceptible, Exposed and Infected individuals that becomes Recovered due to a removal to a quarantine.
#
# ### Remarks for the models units
#
# All sub-population variables (S, I, R, etc) are dimensionless. To obtain the variables, we have to consider that
#
# \begin{align*}
#     &S := \frac{\mathcal{S}}{N} \\
#     &E := \frac{\mathcal{E}}{N} \\
#     &I := \frac{\mathcal{I}}{N} \\
#     &R := \frac{\mathcal{R}}{N} \\
#     &D := \frac{\mathcal{D}}{N} \\
# \end{align*}
#
# with $N$ denoting the total population and $\mathcal{S}$, $\mathcal{E}$, $\mathcal{I}$, $\mathcal{R}$ and $\mathcal{D}$ as the absolute sub-population amounts. Therefore, S, E, I, R and D are given as fractions of the total population.

# %% [markdown]
# Getting population for each country:

# %%
df_population = pd.read_csv(f"{DATA_PATH}/countries of the world.csv")

df_population


# %%
brazil_population = float(df_population[df_population.Country == "Brazil "].Population)
brazil_population = float(210147125)
italy_population = float(df_population[df_population.Country == "Italy "].Population)
china_population = float(df_population[df_population.Country == "China "].Population)
hubei_population = float(58500000)  # from wikipedia!
spain_population = float(df_population[df_population.Country == "Spain "].Population)
iran_population = float(df_population[df_population.Country == "Iran "].Population)
us_population = float(df_population[df_population.Country == "United States "].Population)

target_population = brazil_population
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

df_target_country = df_brazil_cases_by_day
E0, A0, I0, P0, R0, D0, C0, H0 = (
    int(5 * float(df_target_country.confirmed[0])),
    int(1.8 * float(df_target_country.confirmed[0])),
    int(1.2 * float(df_target_country.confirmed[0])),
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
    beta = beta0 * np.exp(-beta1 * t)
    mu = mu0 * np.exp(-mu1 * t)
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
    mu0=1e-7,
    gamma_I=0.1,
    gamma_A=0.15,
    gamma_P=0.14,
    d_I=0.0105,
    d_P=0.003,
    epsilon_I=1 / 3,
    rho=0.88,
    omega=1 / 10,
    sigma=1 / 6.5,
    eta=0,
    beta1=0,
    mu1=0,
    N=1,
):
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
        residual2 = f_exp2 - simulated_qoi2
        residual3 = f_exp3 - simulated_qoi3
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
    (0, 1e-5),  # mu
    (1 / 21, 1 / 14),  # gamma_I
    (1 / 21, 1 / 14),  # gamma_A
    (1 / 21, 1 / 14),  # gamma_P
    (1e-5, 0.1),  # d_I
    (1e-5, 0.1),  # d_P (according to Imperial College report)
    (0, 0.2),  # epsilon_I
    (0.65, 0.9),  # rho
    (0, 1),  # omega
    (1 / 7.5, 1 / 6.5),  # sigma
    (1, 5 * P0),  # E0
    (1, 5 * P0),  # A0
    (1, 20 * P0),  # I0
]
y0_seirpdq_known = S0, P0, R0, D0, C0, H0
# bounds_seirdaq = [(0, 1e-2), (0, 1), (0, 1), (0, 0.2), (0, 0.2), (0, 0.2)]

result_seirpdq = optimize.differential_evolution(
    # seirpdq_least_squares_error_ode,
    seirpdq_least_squares_error_ode_y0,
    bounds=bounds_seirpdq,
    # args=(data_time, [infected_individuals, dead_individuals], seirpdq_ode_solver, y0_seirpdq),
    args=(
        data_time,
        [infected_individuals, dead_individuals, confirmed_cases, recovered_cases],
        seirpdq_ode_solver,
        y0_seirpdq_known,
        target_population,
    ),
    popsize=30,
    strategy="best1bin",
    tol=1e-5,
    recombination=0.95,
    mutation=0.6,
    maxiter=10000,
    polish=True,
    disp=True,
    seed=seed,
    callback=callback_de,
    workers=-1,
)

print(result_seirpdq)

# %%
estimated_y0 = result_seirpdq.x[-3:]
E0, A0, I0 = estimated_y0
y0_seirpdq = S0, int(E0), int(A0), int(I0), P0, R0, D0, C0, H0
print(f"-- Initial conditions: {y0_seirpdq}")

# %%
(
    beta_deterministic,
    mu_deterministic,
    gamma_I_deterministic,
    gamma_A_deterministic,
    gamma_P_deterministic,
    d_I_deterministic,
    d_D_deterministic,
    epsilon_I_deterministic,
    rho_deterministic,
    omega_deterministic,
    sigma_deterministic,
) = result_seirpdq.x[:-3]

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
    mu_deterministic,
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

solution_ODE_seirpdq = seirpdq_ode_solver(y0_seirpdq, (t0, tf), data_time, *result_seirpdq.x[:-3])
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
    r"$\mu$": mu_deterministic,
    r"$\gamma_I$": gamma_I_deterministic,
    r"$\gamma_A$": gamma_A_deterministic,
    r"$\gamma_P$": gamma_P_deterministic,
    r"$d_I$": d_I_deterministic,
    r"$d_P$": d_D_deterministic,
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
    y0_seirpdq, (t0, tf), time_range, *result_seirpdq.x[:-3]
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
        t.dscalar,  # mu
        t.dscalar,  # gamma_I
        t.dscalar,  # gamma_A
        t.dscalar,  # gamma_P
        t.dscalar,  # d_I
        t.dscalar,  # d_D
        t.dscalar,  # epsilon_I
        t.dscalar,  # rho
        t.dscalar,  # omega
        t.dscalar,  # sigma
        t.dscalar,  # E0
        t.dscalar,  # A0
        t.dscalar,  # I0
    ],
    otypes=[t.dmatrix],
)
def seirpdq_ode_wrapper_with_y0(
    time_exp,
    initial_conditions,
    total_population,
    beta,
    mu,
    gamma_I,
    gamma_A,
    gamma_P,
    d_I,
    d_D,
    epsilon_I,
    rho,
    omega,
    sigma,
    E0,
    A0,
    I0,
):
    time_span = (time_exp.min(), time_exp.max())

    args = [beta, mu, gamma_I, gamma_A, gamma_P, d_I, d_D, epsilon_I, rho, omega, sigma]
    S0 = total_population - (
        E0 + A0 + I0 + initial_conditions[4] + initial_conditions[5] + initial_conditions[6]
    )
    initial_conditions = (
        S0,
        E0,
        A0,
        I0,
        initial_conditions[4],  # P
        initial_conditions[5],  # R
        initial_conditions[6],  # D
        initial_conditions[7],  # C
        initial_conditions[8],  # H
    )
    y_model = seirpdq_ode_solver(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    (_, _, _, _, _, _, simulated_qoi1, simulated_qoi2, _,) = simulated_ode_solution

    concatenate_simulated_qoi = np.vstack([simulated_qoi1, simulated_qoi2])

    return concatenate_simulated_qoi


# @theano.compile.ops.as_op(itypes=[t.dvector, t.dvector, t.dscalar, t.dscalar, t.dscalar], otypes=[t.dmatrix])
# def seirpdq_ode_wrapper(time_exp, initial_conditions, beta, gamma, delta):
#     time_span = (time_exp.min(), time_exp.max())

#     args = [beta, gamma, delta]
#     y_model = seirpdq_ode_solver(initial_conditions, time_span, time_exp, *args)
#     simulated_time = y_model.t
#     simulated_ode_solution = y_model.y
#     _, _, _, _, simulated_qoi1, simulated_qoi2 = simulated_ode_solution

#     concatenate_simulated_qoi = np.vstack([simulated_qoi1, simulated_qoi2])

#     return concatenate_simulated_qoi


# %%
percent_calibration = 0.95
with pm.Model() as model_mcmc:
    # Prior distributions for the model's parameters
    beta = pm.Uniform(
        "beta",
        lower=(1 - percent_calibration) * beta_deterministic,
        upper=(1 + percent_calibration) * beta_deterministic,
    )
    mu = pm.Uniform(
        "mu",
        lower=(1 - percent_calibration) * mu_deterministic,
        upper=(1 + percent_calibration) * mu_deterministic,
    )
    gamma_I = pm.Uniform(
        "gamma_I",
        lower=(1 - percent_calibration) * gamma_I_deterministic,
        upper=(1 + percent_calibration) * gamma_I_deterministic,
    )
    gamma_A = pm.Uniform(
        "gamma_A",
        lower=(1 - percent_calibration) * gamma_A_deterministic,
        upper=(1 + percent_calibration) * gamma_A_deterministic,
    )
    gamma_P = pm.Uniform(
        "gamma_P",
        lower=(1 - percent_calibration) * gamma_P_deterministic,
        upper=(1 + percent_calibration) * gamma_P_deterministic,
    )
    d_I = pm.Uniform(
        "d_I",
        lower=(1 - percent_calibration) * d_I_deterministic,
        upper=(1 + percent_calibration) * d_I_deterministic,
    )
    d_D = pm.Uniform(
        "d_P",
        lower=(1 - percent_calibration) * d_D_deterministic,
        upper=(1 + percent_calibration) * d_D_deterministic,
    )
    epsilon_I = pm.Uniform(
        "epsilon_I",
        lower=(1 - percent_calibration) * epsilon_I_deterministic,
        upper=(1 + percent_calibration) * epsilon_I_deterministic,
    )
    # rho = pm.Uniform('rho', lower=(1 - percent_calibration) * rho_deterministic, upper=(1 + percent_calibration) * rho_deterministic)
    rho = pm.Uniform("rho", lower=0.65, upper=0.9)
    omega = pm.Uniform(
        "omega",
        lower=(1 - percent_calibration) * omega_deterministic,
        upper=(1 + percent_calibration) * omega_deterministic,
    )
    sigma = pm.Uniform(
        "sigma",
        lower=(1 - percent_calibration) * sigma_deterministic,
        upper=(1 + percent_calibration) * sigma_deterministic,
    )
    # eta = pm.Uniform('eta', lower=(1 - percent_calibration) * eta_deterministic, upper=(1 + percent_calibration) * eta_deterministic)
    E0_uncertain = pm.Uniform(
        "E0", lower=(1 - percent_calibration) * E0, upper=(1 + percent_calibration) * E0
    )
    A0_uncertain = pm.Uniform(
        "A0", lower=(1 - percent_calibration) * A0, upper=(1 + percent_calibration) * A0
    )
    I0_uncertain = pm.Uniform(
        "I0", lower=(1 - percent_calibration) * I0, upper=(1 + percent_calibration) * I0
    )
    standard_deviation = pm.Uniform("std_deviation", lower=5e2, upper=1.5e3)

    # Defining the deterministic formulation of the problem
    fitting_model = pm.Deterministic(
        "seirpdq_model",
        seirpdq_ode_wrapper_with_y0(
            theano.shared(data_time),
            theano.shared(np.array(y0_seirpdq)),
            theano.shared(target_population),
            beta,
            mu,
            gamma_I,
            gamma_A,
            gamma_P,
            d_I,
            d_D,
            epsilon_I,
            rho,
            omega,
            sigma,
            E0_uncertain,
            A0_uncertain,
            I0_uncertain,
        ),
    )

    likelihood_model = pm.Normal(
        "likelihood_model", mu=fitting_model, sigma=standard_deviation, observed=observations_to_fit
    )

    # The Monte Carlo procedure driver
    step = pm.step_methods.DEMetropolis()
    seirdpq_trace_calibration = pm.sample(
        65000, chains=8, cores=8, step=step, random_seed=seed, tune=25000
    )


# %%
plot_step = 100


# %%
pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("beta"))
plt.savefig("seirpdq_beta_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("mu"))
plt.savefig("seirpdq_mu_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("gamma_I"))
plt.savefig("seirpdq_gamma_I_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("gamma_A"))
plt.savefig("seirpdq_gamma_A_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("gamma_P"))
plt.savefig("seirpdq_gamma_P_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("d_I"))
plt.savefig("seirpdq_d_I_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("d_P"))
plt.savefig("seirpdq_d_D_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("epsilon_I"))
plt.savefig("seirpdq_epsilon_I_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("rho"))
plt.savefig("seirpdq_rho_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("omega"))
plt.savefig("seirpdq_omega_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("sigma"))
plt.savefig("seirpdq_sigma_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("E0"))
plt.savefig("seirpdq_E0_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("A0"))
plt.savefig("seirpdq_A0_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("I0"))
plt.savefig("seirpdq_I0_traceplot_cal.png")
plt.show()

pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=("std_deviation"))
plt.savefig("seirpdq_SD_traceplot_cal.png")
plt.show()

# pm.traceplot(seirdpq_trace_calibration[::plot_step], var_names=('omega'))
# plt.savefig('seirpdq_omega_traceplot_cal.png')
# plt.show()


# %%
# pm.plot_posterior(seirdpq_trace_calibration[::plot_step], var_names=('beta'), kind='hist', round_to=5)
# plt.savefig('seirpdq_beta_posterior_cal.png')
# plt.show()

# pm.plot_posterior(seirdpq_trace_calibration[::plot_step], var_names=('gamma'), kind='hist', round_to=5)
# plt.savefig('seirpdq_gamma_posterior_cal.png')
# plt.show()

# pm.plot_posterior(seirdpq_trace_calibration[::plot_step], var_names=('delta'), kind='hist', round_to=5)
# plt.savefig('seirpdq_delta_posterior_cal.png')
# plt.show()

# pm.plot_posterior(seirdpq_trace_calibration[::plot_step], var_names=('omega'), kind='hist', round_to=3)
# plt.savefig('seirpdq_omega_posterior_cal.png')
# plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("beta"), kind="hist", round_to=5
)
plt.savefig("seirpdq_beta_posterior_cal.png")
plt.show()

pm.plot_posterior(seirdpq_trace_calibration[::plot_step], var_names=("mu"), kind="hist", round_to=5)
plt.savefig("seirpdq_mu_posterior_cal.png")
plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("gamma_I"), kind="hist", round_to=5
)
plt.savefig("seirpdq_gamma_I_posterior_cal.png")
plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("gamma_A"), kind="hist", round_to=5
)
plt.savefig("seirpdq_gamma_A_posterior_cal.png")
plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("gamma_P"), kind="hist", round_to=5
)
plt.savefig("seirpdq_gamma_P_posterior_cal.png")
plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("d_I"), kind="hist", round_to=5
)
plt.savefig("seirpdq_d_I_posterior_cal.png")
plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("d_P"), kind="hist", round_to=5
)
plt.savefig("seirpdq_d_D_posterior_cal.png")
plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("epsilon_I"), kind="hist", round_to=5
)
plt.savefig("seirpdq_epsilon_I_posterior_cal.png")
plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("rho"), kind="hist", round_to=5
)
plt.savefig("seirpdq_rho_posterior_cal.png")
plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("omega"), kind="hist", round_to=5
)
plt.savefig("seirpdq_omega_posterior_cal.png")
plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("sigma"), kind="hist", round_to=5
)
plt.savefig("seirpdq_sigma_posterior_cal.png")
plt.show()

pm.plot_posterior(seirdpq_trace_calibration[::plot_step], var_names=("E0"), kind="hist", round_to=5)
plt.savefig("seirpdq_E0_posterior_cal.png")
plt.show()

pm.plot_posterior(seirdpq_trace_calibration[::plot_step], var_names=("A0"), kind="hist", round_to=5)
plt.savefig("seirpdq_A0_posterior_cal.png")
plt.show()

pm.plot_posterior(seirdpq_trace_calibration[::plot_step], var_names=("I0"), kind="hist", round_to=5)
plt.savefig("seirpdq_I0_posterior_cal.png")
plt.show()

pm.plot_posterior(
    seirdpq_trace_calibration[::plot_step], var_names=("std_deviation"), kind="hist", round_to=5
)
plt.savefig("seirpdq_SD_posterior_cal.png")
plt.show()

# %%
percentile_cut = 5

y_min = np.percentile(seirdpq_trace_calibration["seirpdq_model"], percentile_cut, axis=0)
y_max = np.percentile(seirdpq_trace_calibration["seirpdq_model"], 100 - percentile_cut, axis=0)
y_fit = np.percentile(seirdpq_trace_calibration["seirpdq_model"], 50, axis=0)


# %%
std_deviation = seirdpq_trace_calibration.get_values("std_deviation")
sd_pop = np.sqrt(std_deviation.mean())
print(f"-- Estimated standard deviation mean: {sd_pop}")
sd_pop


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
plt.show()

# %% [markdown]
# Now we evaluate prediction. We have to retrieve parameter realizations.

# %%
print("\n*** Performing Bayesian prediction ***")
dict_realizations = dict()

print("-- Exporting calibrated parameter to CSV")

start_time = time.time()

beta_prediction = seirdpq_trace_calibration.get_values("beta")
dict_realizations["beta"] = beta_prediction

mu_prediction = seirdpq_trace_calibration.get_values("mu")
dict_realizations["mu"] = mu_prediction

gamma_I_prediction = seirdpq_trace_calibration.get_values("gamma_I")
dict_realizations["gamma_I"] = gamma_I_prediction

gamma_A_prediction = seirdpq_trace_calibration.get_values("gamma_A")
dict_realizations["gamma_A"] = gamma_A_prediction

gamma_P_prediction = seirdpq_trace_calibration.get_values("gamma_P")
dict_realizations["gamma_P"] = gamma_P_prediction

d_I_prediction = seirdpq_trace_calibration.get_values("d_I")
dict_realizations["d_I"] = d_I_prediction

d_D_prediction = seirdpq_trace_calibration.get_values("d_P")
dict_realizations["d_D"] = d_D_prediction

epsilon_I_prediction = seirdpq_trace_calibration.get_values("epsilon_I")
dict_realizations["epsilon_I"] = epsilon_I_prediction

rho_prediction = seirdpq_trace_calibration.get_values("rho")
dict_realizations["rho"] = rho_prediction

omega_prediction = seirdpq_trace_calibration.get_values("omega")
dict_realizations["omega"] = omega_prediction

sigma_prediction = seirdpq_trace_calibration.get_values("sigma")
dict_realizations["sigma"] = sigma_prediction

# eta_prediction = seirdpq_trace_calibration.get_values("eta")
# dict_realizations["eta"] = eta_prediction

E0_prediction = seirdpq_trace_calibration.get_values("E0")
dict_realizations["E0"] = E0_prediction

A0_prediction = seirdpq_trace_calibration.get_values("A0")
dict_realizations["A0"] = A0_prediction

I0_prediction = seirdpq_trace_calibration.get_values("I0")
dict_realizations["I0"] = I0_prediction

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
number_of_total_realizations = len(beta_prediction)
for realization in trange(number_of_total_realizations):
    parameters_realization = [
        beta_prediction[realization],
        mu_prediction[realization],
        gamma_I_prediction[realization],
        gamma_A_prediction[realization],
        gamma_P_prediction[realization],
        d_I_prediction[realization],
        d_D_prediction[realization],
        epsilon_I_prediction[realization],
        rho_prediction[realization],
        omega_prediction[realization],
        sigma_prediction[realization],
    ]
    y0_estimated = (
        S0,
        E0_prediction[realization],
        A0_prediction[realization],
        I0_prediction[realization],
        P0,
        R0,
        D0,
        C0,
        H0,
    )
    solution_ODE_predict = seirpdq_ode_solver(
        y0_estimated, (t0, tf), time_range, *parameters_realization
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
