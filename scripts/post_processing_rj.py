# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # COVID-19 SEIRPD-Q: Forward UQ Propagation
# %% [markdown]
# ## Imports

# %%
import os
import time
import matplotlib.pyplot as plt
import arviz as az
from arviz.utils import Numba
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as t
from numba import jit
from scipy import optimize
from scipy.stats import gaussian_kde 
from scipy.integrate import solve_ivp
from tqdm import tqdm, trange
from tqdm.autonotebook import tqdm, trange
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Initial Settings

# %%
# For the sake of reproducibility! :)
seed = 12345
np.random.seed(seed)

# Other Settings
plt.style.use('seaborn-talk')
THEANO_FLAGS = 'optimizer=fast_compile'
Numba.enable_numba()

#Paths
DATA_PATH = '../pydemic/data'
OUTPUT_PATH = './output_post_processing/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Matplotlib
ALPHA = 0.2
MARKER_SIZE = 5
ACOLOR = 'orange'
ICOLOR = 'green'
PCOLOR = 'blue'
DCOLOR = 'red'
CCOLOR = 'indigo'
RTCOLOR = 'black'
PEAK_COLOR = 'darkblue'
DATA_CCOLOR = 'plum'
DATA_DCOLOR = 'rosybrown'

# %% [markdown]
# ## Population

# %%
# Gathered: IBGE 2019
brazil_population = float(210147125)
rio_population = float(6718903)
sp_state_population = float(45919049)
rj_state_population = float(17264943)
ce_state_population = float(9132078)

# %% [markdown]
# ## Read Datasets

# %%
# Dataset: Brazil
df_brazil_states_cases = pd.read_csv(
    f'{DATA_PATH}/covid19br/cases-brazil-states.csv',
    usecols=['date', 'state', 'totalCases', 'deaths', 'recovered'],
    parse_dates=['date'],
)
df_brazil_states_cases.fillna(value={'recovered': 0}, inplace=True)
df_brazil_states_cases = df_brazil_states_cases[df_brazil_states_cases.state != 'TOTAL']

# Import Data (Confirmed > 5)
def get_brazil_state_dataframe(df_brazil: pd.DataFrame, state_name: str, confirmed_lower_threshold: int = 5) -> pd.DataFrame:
    df_brazil = df_brazil.copy()
    df_state_cases = df_brazil[df_brazil.state == state_name]
    df_state_cases.reset_index(inplace=True)
    columns_rename = {'totalCases': 'confirmed'}
    df_state_cases.rename(columns=columns_rename, inplace=True)
    df_state_cases['active'] = (df_state_cases['confirmed'] - df_state_cases['deaths'] - df_state_cases['recovered'])

    df_state_cases = df_state_cases[df_state_cases.confirmed > confirmed_lower_threshold]
    day_range_list = list(range(len(df_state_cases.confirmed)))
    df_state_cases['day'] = day_range_list
    return df_state_cases

# States: SP, RJ, CE
df_sp_state_cases = get_brazil_state_dataframe(df_brazil_states_cases, state_name='SP')
df_rj_state_cases = get_brazil_state_dataframe(df_brazil_states_cases, state_name='RJ')
df_ce_state_cases = get_brazil_state_dataframe(df_brazil_states_cases, state_name='CE')

# Country: BR
df_brazil_cases_by_day = pd.read_csv(f'{DATA_PATH}/brazil_by_day.csv', parse_dates=['date'])
df_brazil_cases_by_day = df_brazil_cases_by_day[df_brazil_cases_by_day.confirmed > 5]
df_brazil_cases_by_day = df_brazil_cases_by_day.reset_index(drop=True)
df_brazil_cases_by_day['day'] = df_brazil_cases_by_day.date.apply(lambda x: (x - df_brazil_cases_by_day.date.min()).days)

# City: RJ
df_rio_cases_by_day = pd.read_csv(f'{DATA_PATH}/rio_covid19.csv')
df_rio_cases_by_day['active'] = (df_rio_cases_by_day['cases'] - df_rio_cases_by_day['deaths'] - df_rio_cases_by_day['recoveries'])
rio_columns_rename = {'cases': 'confirmed', 'recoveries': 'recovered'}
df_rio_cases_by_day.rename(columns=rio_columns_rename, inplace=True)

# %% [markdown]
# ## <span style="color:red">Target Population: Modify!</span>

# %%
# Target Dataset (BR)
#df_target_country = df_brazil_cases_by_day
#target_population = brazil_population

# Target Dataset (RJ)
df_target_country = df_rj_state_cases
target_population = rj_state_population

# Show Dataset
df_target_country.tail(10)

# %% [markdown]
# ## <span style="color:red">Realization Path: Modify!</span>

# %% [markdown]
# ## Initial Conditions

# %%
# Initial Conditions
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

# SEIRPDQ IC Array (NOT FULLY USED)
y0_seirpdq = S0, E0, A0, I0, P0, R0, D0, C0, H0
print(f'INITIAL CONDITIONS (E0, A0, I0, P0, R0, D0, C0, H0) = {y0_seirpdq}')

# %% [markdown]
# ## Extra Functions

# %%
# Reproduction Number
def calculate_reproduction_number(S0, beta, mu, gamma_A, gamma_I, d_I, epsilon_I, rho, omega, sigma=1/7):
    left_term = sigma*(1 - rho)*mu/((sigma + omega)*(gamma_A + omega))
    right_term = beta*sigma*rho/((sigma + omega)*(gamma_I + d_I + omega + epsilon_I))
    return (left_term + right_term)*S0

# %% [markdown]
# ### Exponential Decay
# We can conveniently define an exponential decay with a given half-life time ($t_{1/2}$) in relation to time $t_0$, where exponential decay begins:
# 
# \begin{equation}
# \theta(t; t_0, \theta_0, t_{1/2}) := \theta_0 2^{-\lambda(t - t_0)}
# \end{equation}
# 
# \begin{equation}
# \lambda := \dfrac{\ln{2}}{t_{1/2}}
# \end{equation}

# %%
@jit(nopython=True)
def exp_vanishing(t, value_1, t_transition, half_life_time=1e5):
    decay_constant = np.log(2)/half_life_time
    return np.where(t < t_transition, value_1, value_1 * np.exp(-decay_constant * (t - t_transition)))
omega_decay_function = exp_vanishing

# %% [markdown]
# ## SEIRPDQ Model

# %%
# Base Settings (NOT REAL PARAMETERS)
@jit(nopython=True)
def seirpdq_model(
    t,
    X,
    beta0=0,
    beta1=0,
    mu0=0,
    mu1=0,
    gamma_I=0,
    gamma_A=0,
    gamma_P=0,
    d_I=0,
    d_P=0,
    omega=0,
    epsilon_I=0,
    rho=0,
    eta=0,
    sigma=0,
    N=0,
    decay_function=omega_decay_function,
    transition=0,
    half_life=0
):
    # SEIRPD-Q Python Implementation
    S, E, A, I, P, R, D, C, H = X
    beta = beta0
    mu = mu0
    omega = decay_function(t, omega, transition, half_life) # DECAY
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

# ODE Solvers Wrapper (IF NOT DEFINED, BASE PARAMETERS ARE THESE)
def seirpdq_ode_solver(
    y0,
    t_span,
    t_eval,
    beta0=1e-7,     # CALIBRATED (MODIFY EXTERNALLY)
    omega=1/10,     # CALIBRATED (MODIFY EXTERNALLY)
    d_P=9e-3,       # CALIBRATED (MODIFY EXTERNALLY)
    d_I=2e-4,       # CALIBRATED (MODIFY EXTERNALLY)
    gamma_P=1/14,
    mu0=1e-7,       # CALIBRATED (MODIFY EXTERNALLY)
    gamma_I=1/14,
    gamma_A=1/14,
    epsilon_I=1/3,
    rho=0.85,
    sigma=1/5,
    eta=0,
    beta1=0,
    mu1=0,
    N=1,
    decay_function=omega_decay_function,
    transition=100,
    half_life=1e5   # MODIFY EXTERNALLY
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
            decay_function=decay_function,
            transition=transition,
            half_life=half_life
        ),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='LSODA',
    )

    return solution_ODE

# %% [markdown]
# ## Parameter Realizations

# %%
# Base File
path_to_realizations = f'./calibration_realizations.csv'
parameters_realizations = pd.read_csv(path_to_realizations)
parameters_realizations.head(10)


# %%
# Calibrated Parameters
beta_realizations = parameters_realizations.beta.values
omega_realizations = parameters_realizations.omega.values
d_I_realizations = parameters_realizations.d_I.values
d_P_realizations = parameters_realizations.d_P.values

# %% [markdown]
# ## Frequency Plots

# %%
# *------------*
# | Plot: beta |
# *------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.hist(beta_realizations, bins='doane', rwidth=0.9)
plt.xlabel(r'$\beta$')
plt.ylabel('Frequency')

ax.set_xlim([4.4e-8, 5.0e-8])
ax.set_ylim([0,800])
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_frequency_beta.png')

# *-------------*
# | Plot: omega |
# *-------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.hist(omega_realizations, bins='doane', rwidth=0.9)
plt.xlabel(r'$\omega$')
plt.ylabel('Frequency')

ax.set_xlim([0.012,0.017])
ax.set_ylim([0,700])
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_frequency_omega.png')

# *----------*
# | Plot: dI |
# *----------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.hist(d_I_realizations, bins='doane', rwidth=0.9)
plt.xlabel(r'$d_I$')
plt.ylabel('Frequency')

ax.set_xlim([0.00,0.010])
ax.set_ylim([0,800])
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_frequency_dI.png')

# *----------*
# | Plot: dP |
# *----------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.hist(d_P_realizations, bins='doane', rwidth=0.9)
plt.xlabel(r'$d_P$')
plt.ylabel('Frequency')

ax.set_xlim([0.009,0.014])
ax.set_ylim([0,800])
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_frequency_dP.png')

# %% [markdown]
# ## <span style="color:red">Defining Base Parameters: Modify!</span> 

# %%
# Bayesian Calibration Parameters (BR)
# beta_deterministic = 5.965935e-09
# omega_deterministic = 1.970400e-02
# gamma_I_deterministic = 1/14
# gamma_A_deterministic = 1/14
# gamma_P_deterministic = 1/14
# d_I_deterministic = 1.356770e-02
# d_P_deterministic = 4.168171e-03
# epsilon_I_deterministic = 1/3
# rho_deterministic = 0.85
# sigma_deterministic = 1/5
# eta_deterministic = 0

# Bayesian Calibration Parameters (RJ)
beta_deterministic = 4.658646e-08
omega_deterministic = 1.438746e-02
gamma_I_deterministic = 1/14
gamma_A_deterministic = 1/14
gamma_P_deterministic = 1/14
d_I_deterministic = 5.549912e-04
d_P_deterministic = 1.315661e-02
epsilon_I_deterministic = 1/3
rho_deterministic = 0.85
sigma_deterministic = 1/5
eta_deterministic = 0

# %% [markdown]
# ## <span style="color:red">Solver Settings: Modify!</span>

# %%
# Dataset Cases
data_time = df_target_country.day.values.astype(np.float64)
infected_individuals = df_target_country.active.values
dead_individuals = df_target_country.deaths.values
confirmed_cases = df_target_country.confirmed.values
recovered_cases = df_target_country.recovered.values

# Solver Settings
t0 = float(data_time.min())
number_of_days_after_last_record = 200 # MODIFY HERE
tf = data_time.max() + number_of_days_after_last_record
time_range = np.linspace(t0, tf, int(tf - t0) + 1)
print(f'Last Simulation Day: {tf}')

# %% [markdown]
# ## Forward Simulation

# %%
S_predicted = list()
E_predicted = list()
A_predicted = list()
I_predicted = list()
P_predicted = list()
R_predicted = list()
D_predicted = list()
C_predicted = list()
H_predicted = list()
Rt_predicted = list()

number_of_total_realizations = beta_realizations.shape[0]
for realization in trange(number_of_total_realizations, desc='Performing Realizations'):
    # Set Parameters
    kwargs_parameters = {
        'beta0': beta_realizations[realization],
        'mu0': beta_realizations[realization],
        'omega': omega_realizations[realization],
        'd_I': d_I_realizations[realization],
        'd_P': d_P_realizations[realization],
        'transition': 100, # MODIFY HERE
        'half_life': 1e5   # MODIFY HERE
    }
    
    # Solve ODE System: SEIRDAQ
    solution_ODE_predict = seirpdq_ode_solver(y0_seirpdq, (t0, tf), time_range, **kwargs_parameters)
    
    # Time & Solutions
    t_computed_predict, y_computed_predict = solution_ODE_predict.t, solution_ODE_predict.y
    S, E, A, I, P, R, D, C, H = y_computed_predict

    # Calculate R(t)
    reproduction_number_t = calculate_reproduction_number(
        S,
        beta_realizations[realization],
        beta_realizations[realization],
        gamma_I_deterministic,
        gamma_A_deterministic,
        d_I_realizations[realization],
        epsilon_I_deterministic,
        rho_deterministic,
        omega_realizations[realization],
        sigma_deterministic,
    )

    # Append Solutions
    S_predicted.append(S)
    E_predicted.append(E)
    A_predicted.append(A)
    I_predicted.append(I)
    P_predicted.append(P)
    R_predicted.append(R)
    D_predicted.append(D)
    C_predicted.append(C)
    H_predicted.append(H)
    Rt_predicted.append(reproduction_number_t)

# Separate Solutions
S_predicted = np.array(S_predicted)
E_predicted = np.array(E_predicted)
A_predicted = np.array(A_predicted)
I_predicted = np.array(I_predicted)
P_predicted = np.array(P_predicted)
R_predicted = np.array(R_predicted)
D_predicted = np.array(D_predicted)
C_predicted = np.array(C_predicted)
H_predicted = np.array(H_predicted)
Rt_predicted = np.array(Rt_predicted)

# Percentiles
percentile_cut = 2.5

# Min, Max & Mean: Confirmed
C_min = np.percentile(C_predicted, percentile_cut, axis=0)
C_max = np.percentile(C_predicted, 100 - percentile_cut, axis=0)
C_mean = np.percentile(C_predicted, 50, axis=0)

# Min, Max & Mean: Positively Diagnosed
P_min = np.percentile(P_predicted, percentile_cut, axis=0)
P_max = np.percentile(P_predicted, 100 - percentile_cut, axis=0)
P_mean = np.percentile(P_predicted, 50, axis=0)

# Min, Max & Mean: Infected
I_min = np.percentile(I_predicted, percentile_cut, axis=0)
I_max = np.percentile(I_predicted, 100 - percentile_cut, axis=0)
I_mean = np.percentile(I_predicted, 50, axis=0)

# Min, Max & Mean: Asymptomatic
A_min = np.percentile(A_predicted, percentile_cut, axis=0)
A_max = np.percentile(A_predicted, 100 - percentile_cut, axis=0)
A_mean = np.percentile(A_predicted, 50, axis=0)

# Min, Max & Mean: Dead
D_min = np.percentile(D_predicted, percentile_cut, axis=0)
D_max = np.percentile(D_predicted, 100 - percentile_cut, axis=0)
D_mean = np.percentile(D_predicted, 50, axis=0)

# Min, Max & Mean: R(t)
Rt_min = np.percentile(Rt_predicted, percentile_cut, axis=0)
Rt_max = np.percentile(Rt_predicted, 100 - percentile_cut, axis=0)
Rt_mean = np.percentile(Rt_predicted, 50, axis=0)

# Min, Max & Mean: Peak Day
peak_day_min = np.argmax(P_min)
peak_day_max = np.argmax(P_max)
peak_day_mean = np.argmax(P_mean)

# Min, Max & Mean: Positively Diagnosed - Peak Day
min_diagnosed_peak = int(P_min[peak_day_min])
mean_diagnosed_peak = int(P_mean[peak_day_mean])
max_diagnosed_peak = int(P_max[peak_day_max])


# %%
# Summary Results
print(100*'-')
print(f'-- Min. diagnosed peak day: {peak_day_min}')
print(f'-- Mean diagnosed peak day: {peak_day_mean}')
print(f'-- Max. diagnosed peak day: {peak_day_max}')
print(f'-- Min. number of diagnosed at peak day: {min_diagnosed_peak}')
print(f'-- Mean number of diagnosed at peak day: {mean_diagnosed_peak}')
print(f'-- Max. number of diagnosed at peak day: {max_diagnosed_peak}\n')

print(f'-- Min. number of cases: {int(C_min[-1])}')
print(f'-- Mean number of cases: {int(C_mean[-1])}')
print(f'-- Max. number of cases: {int(C_max[-1])}\n')

print(f'-- Min. number of deaths: {int(D_min[-1])}')
print(f'-- Mean number of deaths: {int(D_mean[-1])}')
print(f'-- Max. number of deaths: {int(D_max[-1])}')
print(100*'-')

# %% [markdown]
# ## Plots: Original (Bayes Values)

# %%
# *-------------------------*
# | Plot: Model Predictions |
# *-------------------------*
fig, ax1 = plt.subplots(figsize=(9,7), dpi=300)
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('P(t), I(t), A(t), D(t)')

ax1.plot(t_computed_predict, P_mean, label='Positively Diagnosed (P)', color=PCOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, I_mean, label='Symptomatic Infected (I)', color=ICOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, A_mean, label='Asymptomatic Infected (A)', color=ACOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, D_mean, label='Dead (D)', marker='o', color=DCOLOR, markersize=MARKER_SIZE)
ax1.fill_between(t_computed_predict, P_min, P_max, color=PCOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, I_min, I_max, color=ICOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, A_min, A_max, color=ACOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, D_min, D_max, color=DCOLOR, alpha=ALPHA)
ax1.axvline(x=peak_day_mean, color=PEAK_COLOR, linestyle='-', linewidth=1.7)
ax1.axvline(x=peak_day_min, color=PEAK_COLOR, linestyle='--', linewidth=1.7)
ax1.axvline(x=peak_day_max, color=PEAK_COLOR, linestyle='--', linewidth=1.7)
ax1.plot(data_time, dead_individuals, label='Dead Data', color=DATA_DCOLOR, marker='o', linestyle='', markersize=MARKER_SIZE)
ax1.set_xlim([0,275])
ax1.set_ylim([0,9000])
ax1.xaxis.set_major_locator(MultipleLocator(25))
ax1.yaxis.set_major_locator(MultipleLocator(1000))

ax2 = ax1.twinx()
ax2.set_ylabel('C(t)')
ax2.plot(t_computed_predict, C_mean, label='Confirmed (C)', color=CCOLOR, marker='o', markersize=MARKER_SIZE)
ax2.fill_between(t_computed_predict, C_min, C_max, color=CCOLOR, alpha=ALPHA)
ax2.plot(data_time, confirmed_cases, label='Confirmed Data', color=DATA_CCOLOR, marker='o', linestyle='', markersize=MARKER_SIZE)
ax2.set_ylim([0,100000])
ax2.yaxis.set_major_locator(MultipleLocator(10000))

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()

# Change Label Order
handler1[4], handler2[0] = handler2[0], handler1[4] # CHANGE
label1[4], label2[0] = label2[0], label1[4]         # CHANGE
fig.legend(handler1+handler2, label1+label2, bbox_to_anchor=(0.155, 1.05, 0.7, 0.1), loc='top left', ncol=2, borderaxespad=0.0)

ax1.set_axisbelow(True)
ax1.xaxis.grid(alpha=ALPHA)
ax1.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_bayes.png', bbox_inches='tight')

# %% [markdown]
# #### When $R(t) \leq 1$, the disease is not likely to spread anymore. Let's estimate this day.

# %%
# Estimation
Rt_near_one_threshold = Rt_predicted - 1
num_of_realizations = Rt_near_one_threshold.shape[0]
control_day_record = list()
for realization in range(num_of_realizations):
    Rt_realization_near_one = Rt_near_one_threshold[realization, :]
    Rt_realization_near_one = Rt_realization_near_one[Rt_realization_near_one > 0]
    control_day = len(Rt_realization_near_one) - 1
    control_day_record.append(control_day)
    
control_day_record = np.array(control_day_record)
control_day_low_percentile = np.percentile(control_day_record, percentile_cut)
control_day_high_percentile = np.percentile(control_day_record, 100 - percentile_cut)

# *------------------*
# | Plot: R(t) Curve |
# *------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.plot(t_computed_predict, Rt_mean, color=RTCOLOR, marker='o', markersize=MARKER_SIZE)
plt.fill_between(t_computed_predict, Rt_min, Rt_max, color=CCOLOR, alpha=ALPHA)
plt.axvline(x=control_day_low_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)
plt.axvline(x=control_day_high_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)

plt.xlabel('Time (Days)')
plt.ylabel(r'$\mathcal{R}(t)$')
ax.set_xlim([0,275])
ax.set_ylim([0,3.0])
ax.xaxis.set_major_locator(MultipleLocator(25))
plt.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'Rt_prediction_bayes.png')

# *------------------------------*
# | Plot: R(t) Frequency Diagram |
# *------------------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
az.plot_dist(control_day_record, hist_kwargs={'density': False})
plt.axvline(x=control_day_low_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)
plt.axvline(x=control_day_high_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)

plt.xlabel(r'$\mathcal{R}(t)\leq 1$ Day')
plt.ylabel('Frequency')
ax.set_xlim([62,84])
ax.set_ylim([0,600])
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'Rt_one_dist.png')

# %% [markdown]
# #### Other Quantifications

# %%
# Peak Day Occurence
P_peak_days = np.argmax(P_predicted, axis=1)
P_peak_days_low_percentile = np.percentile(P_peak_days, percentile_cut)
P_peak_days_high_percentile = np.percentile(P_peak_days, 100 - percentile_cut)

# Max. Diagnosed Number
P_max_values = np.max(P_predicted, axis=1)
P_max_values_low_percentile = np.percentile(P_max_values, percentile_cut)
P_max_values_high_percentile = np.percentile(P_max_values, 100 - percentile_cut)

# Distributions: Final Cumulative Confirmed Cases & Deaths
C_last_low_percentile = np.percentile(C_predicted[:, -1], percentile_cut)
C_last_high_percentile = np.percentile(C_predicted[:, -1], 100 - percentile_cut)
D_last_low_percentile = np.percentile(D_predicted[:, -1], percentile_cut)
D_last_high_percentile = np.percentile(D_predicted[:, -1], 100 - percentile_cut)

# *---------------------------*
# | Plot: Peak Day Occurrence |
# *---------------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
az.plot_dist(P_peak_days, hist_kwargs={'density': False})
plt.axvline(x=P_peak_days_low_percentile, color=PEAK_COLOR, linestyle='--', linewidth=1.7)
plt.axvline(x=P_peak_days_high_percentile, color=PEAK_COLOR, linestyle='--', linewidth=1.7)

plt.xlabel('Peak Day')
plt.ylabel('Frequency')
ax.set_xlim([70,92])
ax.set_ylim([0,600])
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'dist_peak.png')

# *-----------------------------*
# | Plot: Max. Diagnosed Number |
# *-----------------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.hist(P_max_values, rwidth=0.9, bins='doane')
plt.axvline(x=P_max_values_low_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)
plt.axvline(x=P_max_values_high_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)

plt.xlabel('Max. Positively Diagnosed')
plt.ylabel('Frequency')
ax.set_xlim([5000,11000])
ax.set_ylim([0,700])
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'dist_P_max.png')

# *-----------------------------*
# | Plot: Total Confirmed Cases |
# *-----------------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.hist(C_predicted[:, -1], rwidth=0.9, bins='doane')
plt.text(0.885 * C_last_low_percentile, 605, f'{C_last_low_percentile:.0f}', fontsize=15, color=DCOLOR)
plt.text(1.008 * C_last_high_percentile, 605, f'{C_last_high_percentile:.0f}', fontsize=15, color=DCOLOR)
plt.axvline(x=C_last_low_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)
plt.axvline(x=C_last_high_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)

plt.xlabel('Total Confirmed Cases')
plt.ylabel('Frequency')
ax.set_xlim([30000,75000])
ax.set_ylim([0,700])
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'dist_C_last.png')

# *--------------------*
# | Plot: Total Deaths |
# *--------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.hist(D_predicted[:, -1], rwidth=0.9, bins='doane')
plt.text(0.890 * D_last_low_percentile, 505, f'{D_last_low_percentile:.0f}', fontsize=15, color=DCOLOR)
plt.text(1.012 * D_last_high_percentile, 505, f'{D_last_high_percentile:.0f}', fontsize=15, color=DCOLOR)
plt.axvline(x=D_last_low_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)
plt.axvline(x=D_last_high_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)

plt.xlabel('Total Dead Cases')
plt.ylabel('Frequency')
ax.set_xlim([4000,12000])
ax.set_ylim([0,700])
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'dist_D_last.png')

# %% [markdown]
# ## Prediction Validation
# #### Prediction based on current date (06/05), one week later (13/05).

# %%
# Prediction Validation
calibration_last_day = df_target_country.day.values[-1]
today_day = calibration_last_day + 7
confirmed_cases_today = 18728
deaths_today = 2050

# *----------------------------------*
# | Plot: Confirmed Cases Validation |
# *----------------------------------*
fig, ax = plt.subplots(figsize=(9,7))
plt.hist(C_predicted[:, today_day], rwidth=0.9, bins='doane')
plt.axvline(x=confirmed_cases_today, color=DCOLOR, linestyle='--', linewidth=1.7)
plt.text(0.967 * confirmed_cases_today, 505, f'{confirmed_cases_today:.0f}', fontsize=15, color=DCOLOR)

plt.xlabel('Total Confirmed Cases')
plt.ylabel('Frequency')
ax.set_ylim([0,600])
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.show()

# *-------------------------*
# | Plot: Deaths Validation |
# *-------------------------*
fig, ax = plt.subplots(figsize=(9,7))
plt.hist(D_predicted[:, today_day], rwidth=0.9, bins='doane')
plt.axvline(x=deaths_today, color=DCOLOR, linestyle='--', linewidth=1.7)
plt.text(0.967 * deaths_today, 605, f'{deaths_today:.0f}', fontsize=15, color=DCOLOR)

plt.xlabel('Total Dead Cases')
plt.ylabel('Frequency')
ax.set_ylim([0,700])
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## New Study: Decay 01

# %%
# New Study (Parameters)
half_life1 = 0.1
transition1 = 100


# %%
S1_predicted = list()
E1_predicted = list()
A1_predicted = list()
I1_predicted = list()
P1_predicted = list()
R1_predicted = list()
D1_predicted = list()
C1_predicted = list()
H1_predicted = list()
Rt1_predicted = list()

number_of_total_realizations = beta_realizations.shape[0]
for realization in trange(number_of_total_realizations, desc='Performing Realizations'):
    # Set Parameters
    kwargs_parameters = {
        'beta0': beta_realizations[realization],
        'mu0': beta_realizations[realization],
        'omega': omega_realizations[realization],
        'd_I': d_I_realizations[realization],
        'd_P': d_P_realizations[realization],
        'transition': transition1, # MODIFY HERE
        'half_life': half_life1    # MODIFY HERE
    }
    
    # Solve ODE System: SEIRDAQ
    solution_ODE_predict = seirpdq_ode_solver(y0_seirpdq, (t0, tf), time_range, **kwargs_parameters)
    
    # Time & Solutions
    t_computed_predict, y_computed_predict = solution_ODE_predict.t, solution_ODE_predict.y
    S1, E1, A1, I1, P1, R1, D1, C1, H1 = y_computed_predict

    # Calculate R(t)
    reproduction_number_t1 = calculate_reproduction_number(
        S1,
        beta_realizations[realization],
        beta_realizations[realization],
        gamma_I_deterministic,
        gamma_A_deterministic,
        d_I_realizations[realization],
        epsilon_I_deterministic,
        rho_deterministic,
        omega_realizations[realization],
        sigma_deterministic,
    )

    # Append Solutions
    S1_predicted.append(S1)
    E1_predicted.append(E1)
    A1_predicted.append(A1)
    I1_predicted.append(I1)
    P1_predicted.append(P1)
    R1_predicted.append(R1)
    D1_predicted.append(D1)
    C1_predicted.append(C1)
    H1_predicted.append(H1)
    Rt1_predicted.append(reproduction_number_t1)

# Separate Solutions
S1_predicted = np.array(S1_predicted)
E1_predicted = np.array(E1_predicted)
A1_predicted = np.array(A1_predicted)
I1_predicted = np.array(I1_predicted)
P1_predicted = np.array(P1_predicted)
R1_predicted = np.array(R1_predicted)
D1_predicted = np.array(D1_predicted)
C1_predicted = np.array(C1_predicted)
H1_predicted = np.array(H1_predicted)
Rt1_predicted = np.array(Rt1_predicted)

# Percentiles
percentile_cut = 2.5

# Min, Max & Mean: Confirmed
C1_min = np.percentile(C1_predicted, percentile_cut, axis=0)
C1_max = np.percentile(C1_predicted, 100 - percentile_cut, axis=0)
C1_mean = np.percentile(C1_predicted, 50, axis=0)

# Min, Max & Mean: Positively Diagnosed
P1_min = np.percentile(P1_predicted, percentile_cut, axis=0)
P1_max = np.percentile(P1_predicted, 100 - percentile_cut, axis=0)
P1_mean = np.percentile(P1_predicted, 50, axis=0)

# Min, Max & Mean: Infected
I1_min = np.percentile(I1_predicted, percentile_cut, axis=0)
I1_max = np.percentile(I1_predicted, 100 - percentile_cut, axis=0)
I1_mean = np.percentile(I1_predicted, 50, axis=0)

# Min, Max & Mean: Asymptomatic
A1_min = np.percentile(A1_predicted, percentile_cut, axis=0)
A1_max = np.percentile(A1_predicted, 100 - percentile_cut, axis=0)
A1_mean = np.percentile(A1_predicted, 50, axis=0)

# Min, Max & Mean: Dead
D1_min = np.percentile(D1_predicted, percentile_cut, axis=0)
D1_max = np.percentile(D1_predicted, 100 - percentile_cut, axis=0)
D1_mean = np.percentile(D1_predicted, 50, axis=0)

# Min, Max & Mean: R(t)
Rt1_min = np.percentile(Rt1_predicted, percentile_cut, axis=0)
Rt1_max = np.percentile(Rt1_predicted, 100 - percentile_cut, axis=0)
Rt1_mean = np.percentile(Rt1_predicted, 50, axis=0)

# Min, Max & Mean: Peak Day
peak1_day_min = np.argmax(P1_min)
peak1_day_max = np.argmax(P1_max)
peak1_day_mean = np.argmax(P1_mean)

# Min, Max & Mean: Positively Diagnosed - Peak Day
min1_diagnosed_peak = int(P1_min[peak1_day_min])
mean1_diagnosed_peak = int(P1_mean[peak1_day_mean])
max1_diagnosed_peak = int(P1_max[peak1_day_max])


# %%
# *-------------*
# | Plot: Decay |
# *-------------*
exp_decay = exp_vanishing(time_range, omega_deterministic, transition1, half_life1)

fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.axvline(x=half_life1 + transition1, color='black', linewidth=1.7)
plt.plot(time_range, exp_decay, color=DCOLOR, marker='o', markersize=MARKER_SIZE, linewidth=1.7)
plt.axhline(omega_deterministic/2, color='black', linewidth=1.7)

plt.xlabel('Time (Days)')
plt.ylabel(r'$\theta$ (${t_{1/2}} = $' + f'{half_life1} Days)')
ax.set_xlim([0,275])
ax.set_ylim([0,0.015])
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(0.003))
ax.set_axisbelow(True)
plt.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'exp_decay1.png')


# %%
# *-------------------------*
# | Plot: Model Predictions |
# *-------------------------*
fig, ax1 = plt.subplots(figsize=(9,7), dpi=300)
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('P(t), I(t), A(t), D(t)')

ax1.plot(t_computed_predict, P1_mean, label='Positively Diagnosed (P)', color=PCOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, I1_mean, label='Symptomatic Infected (I)', color=ICOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, A1_mean, label='Asymptomatic Infected (A)', color=ACOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, D1_mean, label='Dead (D)', marker='o', color=DCOLOR, markersize=MARKER_SIZE)
ax1.fill_between(t_computed_predict, P1_min, P1_max, color=PCOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, I1_min, I1_max, color=ICOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, A1_min, A1_max, color=ACOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, D1_min, D1_max, color=DCOLOR, alpha=ALPHA)
ax1.axvline(x=peak_day_mean, color=PEAK_COLOR, linestyle='-', linewidth=1.7)
ax1.axvline(x=peak_day_min, color=PEAK_COLOR, linestyle='--', linewidth=1.7)
ax1.axvline(x=peak_day_max, color=PEAK_COLOR, linestyle='--', linewidth=1.7)
ax1.plot(data_time, dead_individuals, label='Dead Data', color=DATA_DCOLOR, marker='o', linestyle='', markersize=MARKER_SIZE)
ax.set_xlim([0,275])
ax1.set_ylim([0,13000])
ax1.xaxis.set_major_locator(MultipleLocator(25))
ax1.yaxis.set_major_locator(MultipleLocator(1000))

ax2 = ax1.twinx()
ax2.set_ylabel('C(t)')
ax2.plot(t_computed_predict, C1_mean, label='Confirmed (C)', color=CCOLOR, marker='o', markersize=MARKER_SIZE)
ax2.fill_between(t_computed_predict, C1_min, C1_max, color=CCOLOR, alpha=ALPHA)
ax2.plot(data_time, confirmed_cases, label='Confirmed Data', color=DATA_CCOLOR, marker='o', linestyle='', markersize=MARKER_SIZE)
ax2.set_ylim([0,160000])
ax2.yaxis.set_major_locator(MultipleLocator(10000))

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()

# Change Label Order
handler1[4], handler2[0] = handler2[0], handler1[4] # CHANGE
label1[4], label2[0] = label2[0], label1[4]         # CHANGE
fig.legend(handler1+handler2, label1+label2, bbox_to_anchor=(0.155, 1.05, 0.7, 0.1), loc='top left', ncol=2, borderaxespad=0.0)

#plt.grid()
ax1.set_axisbelow(True)
ax1.xaxis.grid(alpha=ALPHA)
ax1.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_bayes1.png', bbox_inches='tight')

# %% [markdown]
# ## New Study: Decay 02

# %%
# New Study (Parameters)
half_life2 = 15
transition2 = 100


# %%
S2_predicted = list()
E2_predicted = list()
A2_predicted = list()
I2_predicted = list()
P2_predicted = list()
R2_predicted = list()
D2_predicted = list()
C2_predicted = list()
H2_predicted = list()
Rt2_predicted = list()

number_of_total_realizations = beta_realizations.shape[0]
for realization in trange(number_of_total_realizations, desc='Performing Realizations'):
    # Set Parameters
    kwargs_parameters = {
        'beta0': beta_realizations[realization],
        'mu0': beta_realizations[realization],
        'omega': omega_realizations[realization],
        'd_I': d_I_realizations[realization],
        'd_P': d_P_realizations[realization],
        'transition': transition2, # MODIFY HERE
        'half_life': half_life2    # MODIFY HERE
    }
    
    # Solve ODE System: SEIRDAQ
    solution_ODE_predict = seirpdq_ode_solver(y0_seirpdq, (t0, tf), time_range, **kwargs_parameters)
    
    # Time & Solutions
    t_computed_predict, y_computed_predict = solution_ODE_predict.t, solution_ODE_predict.y
    S2, E2, A2, I2, P2, R2, D2, C2, H2 = y_computed_predict

    # Calculate R(t)
    reproduction_number_t2 = calculate_reproduction_number(
        S2,
        beta_realizations[realization],
        beta_realizations[realization],
        gamma_I_deterministic,
        gamma_A_deterministic,
        d_I_realizations[realization],
        epsilon_I_deterministic,
        rho_deterministic,
        omega_realizations[realization],
        sigma_deterministic,
    )

    # Append Solutions
    S2_predicted.append(S2)
    E2_predicted.append(E2)
    A2_predicted.append(A2)
    I2_predicted.append(I2)
    P2_predicted.append(P2)
    R2_predicted.append(R2)
    D2_predicted.append(D2)
    C2_predicted.append(C2)
    H2_predicted.append(H2)
    Rt2_predicted.append(reproduction_number_t2)

# Separate Solutions
S2_predicted = np.array(S2_predicted)
E2_predicted = np.array(E2_predicted)
A2_predicted = np.array(A2_predicted)
I2_predicted = np.array(I2_predicted)
P2_predicted = np.array(P2_predicted)
R2_predicted = np.array(R2_predicted)
D2_predicted = np.array(D2_predicted)
C2_predicted = np.array(C2_predicted)
H2_predicted = np.array(H2_predicted)
Rt2_predicted = np.array(Rt2_predicted)

# Percentiles
percentile_cut = 2.5

# Min, Max & Mean: Confirmed
C2_min = np.percentile(C2_predicted, percentile_cut, axis=0)
C2_max = np.percentile(C2_predicted, 100 - percentile_cut, axis=0)
C2_mean = np.percentile(C2_predicted, 50, axis=0)

# Min, Max & Mean: Positively Diagnosed
P2_min = np.percentile(P2_predicted, percentile_cut, axis=0)
P2_max = np.percentile(P2_predicted, 100 - percentile_cut, axis=0)
P2_mean = np.percentile(P2_predicted, 50, axis=0)

# Min, Max & Mean: Infected
I2_min = np.percentile(I2_predicted, percentile_cut, axis=0)
I2_max = np.percentile(I2_predicted, 100 - percentile_cut, axis=0)
I2_mean = np.percentile(I2_predicted, 50, axis=0)

# Min, Max & Mean: Asymptomatic
A2_min = np.percentile(A2_predicted, percentile_cut, axis=0)
A2_max = np.percentile(A2_predicted, 100 - percentile_cut, axis=0)
A2_mean = np.percentile(A2_predicted, 50, axis=0)

# Min, Max & Mean: Dead
D2_min = np.percentile(D2_predicted, percentile_cut, axis=0)
D2_max = np.percentile(D2_predicted, 100 - percentile_cut, axis=0)
D2_mean = np.percentile(D2_predicted, 50, axis=0)

# Min, Max & Mean: R(t)
Rt2_min = np.percentile(Rt2_predicted, percentile_cut, axis=0)
Rt2_max = np.percentile(Rt2_predicted, 100 - percentile_cut, axis=0)
Rt2_mean = np.percentile(Rt2_predicted, 50, axis=0)

# Min, Max & Mean: Peak Day
peak2_day_min = np.argmax(P2_min)
peak2_day_max = np.argmax(P2_max)
peak2_day_mean = np.argmax(P2_mean)

# Min, Max & Mean: Positively Diagnosed - Peak Day
min2_diagnosed_peak = int(P2_min[peak2_day_min])
mean2_diagnosed_peak = int(P2_mean[peak2_day_mean])
max2_diagnosed_peak = int(P2_max[peak2_day_max])


# %%
# *-------------*
# | Plot: Decay |
# *-------------*
exp_decay = exp_vanishing(time_range, omega_deterministic, transition2, half_life2)

fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.axvline(x=half_life2 + transition2, color='black', linewidth=1.7)
plt.plot(time_range, exp_decay, color=DCOLOR, marker='o', markersize=MARKER_SIZE, linewidth=1.7)
plt.axhline(omega_deterministic/2, color='black', linewidth=1.7)

plt.xlabel('Time (Days)')
plt.ylabel(r'$\theta$ (${t_{1/2}} = $' + f'{half_life2} Days)')
ax.set_xlim([0,275])
ax.set_ylim([0,0.015])
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(0.003))
ax.set_axisbelow(True)
plt.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'exp_decay2.png')


# %%
# *-------------------------*
# | Plot: Model Predictions |
# *-------------------------*
fig, ax1 = plt.subplots(figsize=(9,7), dpi=300)
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('P(t), I(t), A(t), D(t)')

ax1.plot(t_computed_predict, P2_mean, label='Positively Diagnosed (P)', color=PCOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, I2_mean, label='Symptomatic Infected (I)', color=ICOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, A2_mean, label='Asymptomatic Infected (A)', color=ACOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, D2_mean, label='Dead (D)', marker='o', color=DCOLOR, markersize=MARKER_SIZE)
ax1.fill_between(t_computed_predict, P2_min, P2_max, color=PCOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, I2_min, I2_max, color=ICOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, A2_min, A2_max, color=ACOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, D2_min, D2_max, color=DCOLOR, alpha=ALPHA)
ax1.axvline(x=peak_day_mean, color=PEAK_COLOR, linestyle='-', linewidth=1.7)
ax1.axvline(x=peak_day_min, color=PEAK_COLOR, linestyle='--', linewidth=1.7)
ax1.axvline(x=peak_day_max, color=PEAK_COLOR, linestyle='--', linewidth=1.7)
ax1.plot(data_time, dead_individuals, label='Dead Data', color=DATA_DCOLOR, marker='o', linestyle='', markersize=MARKER_SIZE)
ax1.set_xlim([0,275])
ax1.set_ylim([0,10000])
ax1.xaxis.set_major_locator(MultipleLocator(25))
ax1.yaxis.set_major_locator(MultipleLocator(1000))

ax2 = ax1.twinx()
ax2.set_ylabel('C(t)')
ax2.plot(t_computed_predict, C2_mean, label='Confirmed (C)', color=CCOLOR, marker='o', markersize=MARKER_SIZE)
ax2.fill_between(t_computed_predict, C2_min, C2_max, color=CCOLOR, alpha=ALPHA)
ax2.plot(data_time, confirmed_cases, label='Confirmed Data', color=DATA_CCOLOR, marker='o', linestyle='', markersize=MARKER_SIZE)
ax2.set_ylim([0,110000])
ax2.yaxis.set_major_locator(MultipleLocator(10000))

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()

# Change Label Order
handler1[4], handler2[0] = handler2[0], handler1[4] # CHANGE
label1[4], label2[0] = label2[0], label1[4]         # CHANGE
fig.legend(handler1+handler2, label1+label2, bbox_to_anchor=(0.155, 1.05, 0.7, 0.1), loc='top left', ncol=2, borderaxespad=0.0)

#plt.grid()
ax1.set_axisbelow(True)
ax1.xaxis.grid(alpha=ALPHA)
ax1.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_bayes2.png', bbox_inches='tight')

# %% [markdown]
# ## New Study: Decay 03

# %%
# New Study (Parameters)
half_life3 = 20
transition3 = peak_day_mean - 20.0
print(f'New Release Day: {transition3}')


# %%
S3_predicted = list()
E3_predicted = list()
A3_predicted = list()
I3_predicted = list()
P3_predicted = list()
R3_predicted = list()
D3_predicted = list()
C3_predicted = list()
H3_predicted = list()
Rt3_predicted = list()

number_of_total_realizations = beta_realizations.shape[0]
for realization in trange(number_of_total_realizations, desc='Performing Realizations'):
    # Set Parameters
    kwargs_parameters = {
        'beta0': beta_realizations[realization],
        'mu0': beta_realizations[realization],
        'omega': omega_realizations[realization],
        'd_I': d_I_realizations[realization],
        'd_P': d_P_realizations[realization],
        'transition': transition3, # MODIFY HERE
        'half_life': half_life3    # MODIFY HERE
    }
    
    # Solve ODE System: SEIRDAQ
    solution_ODE_predict = seirpdq_ode_solver(y0_seirpdq, (t0, tf), time_range, **kwargs_parameters)
    
    # Time & Solutions
    t_computed_predict, y_computed_predict = solution_ODE_predict.t, solution_ODE_predict.y
    S3, E3, A3, I3, P3, R3, D3, C3, H3 = y_computed_predict

    # Calculate R(t)
    reproduction_number_t3 = calculate_reproduction_number(
        S3,
        beta_realizations[realization],
        beta_realizations[realization],
        gamma_I_deterministic,
        gamma_A_deterministic,
        d_I_realizations[realization],
        epsilon_I_deterministic,
        rho_deterministic,
        omega_realizations[realization],
        sigma_deterministic,
    )

    # Append Solutions
    S3_predicted.append(S3)
    E3_predicted.append(E3)
    A3_predicted.append(A3)
    I3_predicted.append(I3)
    P3_predicted.append(P3)
    R3_predicted.append(R3)
    D3_predicted.append(D3)
    C3_predicted.append(C3)
    H3_predicted.append(H3)
    Rt3_predicted.append(reproduction_number_t3)

# Separate Solutions
S3_predicted = np.array(S3_predicted)
E3_predicted = np.array(E3_predicted)
A3_predicted = np.array(A3_predicted)
I3_predicted = np.array(I3_predicted)
P3_predicted = np.array(P3_predicted)
R3_predicted = np.array(R3_predicted)
D3_predicted = np.array(D3_predicted)
C3_predicted = np.array(C3_predicted)
H3_predicted = np.array(H3_predicted)
Rt3_predicted = np.array(Rt3_predicted)

# Percentiles
percentile_cut = 2.5

# Min, Max & Mean: Confirmed
C3_min = np.percentile(C3_predicted, percentile_cut, axis=0)
C3_max = np.percentile(C3_predicted, 100 - percentile_cut, axis=0)
C3_mean = np.percentile(C3_predicted, 50, axis=0)

# Min, Max & Mean: Positively Diagnosed
P3_min = np.percentile(P3_predicted, percentile_cut, axis=0)
P3_max = np.percentile(P3_predicted, 100 - percentile_cut, axis=0)
P3_mean = np.percentile(P3_predicted, 50, axis=0)

# Min, Max & Mean: Infected
I3_min = np.percentile(I3_predicted, percentile_cut, axis=0)
I3_max = np.percentile(I3_predicted, 100 - percentile_cut, axis=0)
I3_mean = np.percentile(I3_predicted, 50, axis=0)

# Min, Max & Mean: Asymptomatic
A3_min = np.percentile(A3_predicted, percentile_cut, axis=0)
A3_max = np.percentile(A3_predicted, 100 - percentile_cut, axis=0)
A3_mean = np.percentile(A3_predicted, 50, axis=0)

# Min, Max & Mean: Dead
D3_min = np.percentile(D3_predicted, percentile_cut, axis=0)
D3_max = np.percentile(D3_predicted, 100 - percentile_cut, axis=0)
D3_mean = np.percentile(D3_predicted, 50, axis=0)

# Min, Max & Mean: R(t)
Rt3_min = np.percentile(Rt3_predicted, percentile_cut, axis=0)
Rt3_max = np.percentile(Rt3_predicted, 100 - percentile_cut, axis=0)
Rt3_mean = np.percentile(Rt3_predicted, 50, axis=0)

# Min, Max & Mean: Peak Day
peak3_day_min = np.argmax(P3_min)
peak3_day_max = np.argmax(P3_max)
peak3_day_mean = np.argmax(P3_mean)

# Min, Max & Mean: Positively Diagnosed - Peak Day
min3_diagnosed_peak = int(P3_min[peak3_day_min])
mean3_diagnosed_peak = int(P3_mean[peak3_day_mean])
max3_diagnosed_peak = int(P3_max[peak3_day_max])


# %%
# *-------------*
# | Plot: Decay |
# *-------------*
exp_decay = exp_vanishing(time_range, omega_deterministic, transition3, half_life3)

fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.axvline(x=half_life3 + transition3, color='black', linewidth=1.7)
plt.plot(time_range, exp_decay, color=DCOLOR, marker='o', markersize=MARKER_SIZE, linewidth=1.7)
plt.axhline(omega_deterministic/2, color='black', linewidth=1.7)

plt.xlabel('Time (Days)')
plt.ylabel(r'$\theta$ (${t_{1/2}} = $' + f'{half_life3} Days)')
ax.set_xlim([0,275])
ax.set_ylim([0,0.015])
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(0.003))
ax.set_axisbelow(True)
plt.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'exp_decay3.png')


# %%
# *-------------------------*
# | Plot: Model Predictions |
# *-------------------------*
fig, ax1 = plt.subplots(figsize=(9,7), dpi=300)
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('P(t), I(t), A(t), D(t)')

ax1.plot(t_computed_predict, P3_mean, label='Positively Diagnosed (P)', color=PCOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, I3_mean, label='Symptomatic Infected (I)', color=ICOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, A3_mean, label='Asymptomatic Infected (A)', color=ACOLOR, marker='o', markersize=MARKER_SIZE)
ax1.plot(t_computed_predict, D3_mean, label='Dead (D)', marker='o', color=DCOLOR, markersize=MARKER_SIZE)
ax1.fill_between(t_computed_predict, P3_min, P3_max, color=PCOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, I3_min, I3_max, color=ICOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, A3_min, A3_max, color=ACOLOR, alpha=ALPHA)
ax1.fill_between(t_computed_predict, D3_min, D3_max, color=DCOLOR, alpha=ALPHA)
ax1.axvline(x=peak3_day_mean, color=PEAK_COLOR, linestyle='-', linewidth=1.7)
ax1.axvline(x=peak3_day_min, color=PEAK_COLOR, linestyle='--', linewidth=1.7)
ax1.axvline(x=peak3_day_max, color=PEAK_COLOR, linestyle='--', linewidth=1.7)
ax1.plot(data_time, dead_individuals, label='Dead Data', color=DATA_DCOLOR, marker='o', linestyle='', markersize=MARKER_SIZE)
ax1.set_xlim([0,275])
ax1.set_ylim([0,28000])
ax1.xaxis.set_major_locator(MultipleLocator(25))
ax1.yaxis.set_major_locator(MultipleLocator(2000))

ax2 = ax1.twinx()
ax2.set_ylabel('C(t)')
ax2.plot(t_computed_predict, C3_mean, label='Confirmed (C)', color=CCOLOR, marker='o', markersize=MARKER_SIZE)
ax2.fill_between(t_computed_predict, C3_min, C3_max, color=CCOLOR, alpha=ALPHA)
ax2.plot(data_time, confirmed_cases, label='Confirmed Data', color=DATA_CCOLOR, marker='o', linestyle='', markersize=MARKER_SIZE)
ax2.set_ylim([0,400000])

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()

# Change Label Order
handler1[4], handler2[0] = handler2[0], handler1[4] # CHANGE
label1[4], label2[0] = label2[0], label1[4]         # CHANGE
fig.legend(handler1+handler2, label1+label2, bbox_to_anchor=(0.20, 1.05, 0.7, 0.1), loc='top left', ncol=2, borderaxespad=0.0)

#plt.grid()
ax1.set_axisbelow(True)
ax1.xaxis.grid(alpha=ALPHA)
ax1.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_bayes3.png', bbox_inches='tight')

# %% [markdown]
# ## General Plots

# %%
# Summary Results
print('Summary Results (Original):')
print(100*'-')
print(f'-- Peak Day: {peak_day_mean} (CI: {peak_day_min}, {peak_day_max})')
print(f'-- Max. number of confirmed cases estimate (SEIRPD-Q Model): {int(C_mean[-1])} (CI: {np.ceil(C_min[-1])}, {np.ceil(C_max[-1])})')
print(f'-- Max. number of diagnosed individuals (SEIRPD-Q Model): {int(np.max(P_mean))} (CI: {min_diagnosed_peak}, {max_diagnosed_peak})')
print(f'-- Max. number of deaths estimate (SEIRPD-Q Model): {int(D_mean[-1])} (CI: {np.ceil(D_min[-1])}, {np.ceil(D_max[-1])})')
print(100*'-')

# Summary Results
print(f'\nSummary Results (T = {transition1}, HL = {half_life1}):')
print(100*'-')
print(f'-- Peak Day: {peak1_day_mean} (CI: {peak1_day_min}, {peak1_day_max})')
print(f'-- Max. number of confirmed cases estimate (SEIRPD-Q Model): {int(C1_mean[-1])} (CI: {np.ceil(C1_min[-1])}, {np.ceil(C1_max[-1])})')
print(f'-- Max. number of diagnosed individuals (SEIRPD-Q Model): {int(np.max(P1_mean))} (CI: {min1_diagnosed_peak}, {max1_diagnosed_peak})')
print(f'-- Max. number of deaths estimate (SEIRPD-Q Model): {int(D1_mean[-1])} (CI: {np.ceil(D1_min[-1])}, {np.ceil(D1_max[-1])})')
print(100*'-')

# Summary Results
print(f'\nSummary Results (T = {transition2}, HL = {half_life2}):')
print(100*'-')
print(f'-- Peak Day: {peak2_day_mean} (CI: {peak2_day_min}, {peak2_day_max})')
print(f'-- Max. number of confirmed cases estimate (SEIRPD-Q Model): {int(C2_mean[-1])} (CI: {np.ceil(C2_min[-1])}, {np.ceil(C2_max[-1])})')
print(f'-- Max. number of diagnosed individuals (SEIRPD-Q Model): {int(np.max(P2_mean))} (CI: {min2_diagnosed_peak}, {max2_diagnosed_peak})')
print(f'-- Max. number of deaths estimate (SEIRPD-Q Model): {int(D2_mean[-1])} (CI: {np.ceil(D2_min[-1])}, {np.ceil(D2_max[-1])})')
print(100*'-')

# Summary Results
print(f'\nSummary Results (T = {transition3}, HL = {half_life3}):')
print(100*'-')
print(f'-- Peak Day: {peak3_day_mean} (CI: {peak3_day_min}, {peak3_day_max})')
print(f'-- Max. number of confirmed cases estimate (SEIRPD-Q Model): {int(C3_mean[-1])} (CI: {np.ceil(C3_min[-1])}, {np.ceil(C3_max[-1])})')
print(f'-- Max. number of diagnosed individuals (SEIRPD-Q Model): {int(np.max(P3_mean))} (CI: {min3_diagnosed_peak}, {max3_diagnosed_peak})')
print(f'-- Max. number of deaths estimate (SEIRPD-Q Model): {int(D3_mean[-1])} (CI: {np.ceil(D3_min[-1])}, {np.ceil(D3_max[-1])})')
print(100*'-')


# %%
# *-------------------*
# | Plot: R(t) Curves |
# *-------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.plot(t_computed_predict, Rt1_mean, label=r'$\mathcal{R}(t), {t_{1/2}} = $' + f'{half_life1} Days', color='darkgray', marker='o', markersize=MARKER_SIZE)
plt.plot(t_computed_predict, Rt2_mean, label=r'$\mathcal{R}(t), {t_{1/2}} = $' + f'{half_life2} Days', color='dimgray', marker='o', markersize=MARKER_SIZE)
plt.plot(t_computed_predict, Rt_mean, label=r'$\mathcal{R}(t)$', color=RTCOLOR, marker='o', markersize=MARKER_SIZE)
plt.fill_between(t_computed_predict, Rt1_min, Rt1_max, color='darkgray', alpha=ALPHA)
plt.fill_between(t_computed_predict, Rt2_min, Rt2_max, color='dimgray', alpha=ALPHA)
plt.fill_between(t_computed_predict, Rt_min, Rt_max, color=CCOLOR, alpha=ALPHA)
plt.axvline(x=control_day_low_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)
plt.axvline(x=control_day_high_percentile, color=DCOLOR, linestyle='--', linewidth=1.7)

plt.xlabel('Time (Days)')
plt.ylabel(r'$\mathcal{R}(t)$')
plt.legend(loc='best')
ax.set_xlim([0,275])
ax.set_ylim([0,3.0])
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(0.3))
plt.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'Rt_prediction_bayes_all.png')


# %%
# *----------------------*
# | Plot: Mean Confirmed |
# *----------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.plot(t_computed_predict, C1_mean, label=r'Confirmed (C), ${t_{1/2}} = $' + f'{half_life1} Days', color='plum', marker='o', markersize=MARKER_SIZE)
plt.plot(t_computed_predict, C2_mean, label=r'Confirmed (C), ${t_{1/2}} = $' + f'{half_life2} Days', color='mediumpurple', marker='o', markersize=MARKER_SIZE)
plt.plot(t_computed_predict, C_mean, label='Confirmed (C)', color=CCOLOR, marker='o', markersize=MARKER_SIZE)
plt.plot(data_time, confirmed_cases, label='Confirmed Data', color=DATA_CCOLOR, marker='o', linestyle='', markersize=MARKER_SIZE)

plt.xlabel('Time (Days)')
plt.ylabel('C(t)')
plt.legend(loc='lower right')
ax.set_xlim([0,275])
ax.set_ylim([0,60000])
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(5000))
ax.set_axisbelow(True)
plt.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_mean_confirmed.png')

# *-----------------*
# | Plot: Mean Dead |
# *-----------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.plot(t_computed_predict, D1_mean, label=r'Dead (D), ${t_{1/2}} = $' + f'{half_life1} Days', color='pink', marker='o', markersize=MARKER_SIZE)
plt.plot(t_computed_predict, D2_mean, label=r'Dead (D), ${t_{1/2}} = $' + f'{half_life2} Days', color='lightcoral', marker='o', markersize=MARKER_SIZE)
plt.plot(t_computed_predict, D_mean, label='Dead (D)', color=DCOLOR, marker='o', markersize=MARKER_SIZE)
plt.plot(data_time, dead_individuals, label='Dead Data', color=DATA_DCOLOR, marker='o', linestyle='', markersize=MARKER_SIZE)

plt.xlabel('Time (Days)')
plt.ylabel('D(t)')
plt.legend(loc='lower right')
ax.set_xlim([0,275])
ax.set_ylim([0,9000])
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.set_axisbelow(True)
plt.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_mean_dead.png')

# *---------------------------------*
# | Plot: Mean Positively Diagnosed |
# *---------------------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
plt.plot(t_computed_predict, P1_mean, label=r'Positively Diagnosed (P), ${t_{1/2}} = $' + f'{half_life1} Days', color='lightsteelblue', marker='o', markersize=MARKER_SIZE)
plt.plot(t_computed_predict, P2_mean, label=r'Positively Diagnosed (P), ${t_{1/2}} = $' + f'{half_life2} Days', color='royalblue', marker='o', markersize=MARKER_SIZE)
plt.plot(t_computed_predict, P_mean, label='Positively Diagnosed (P)', color=PCOLOR, marker='o', markersize=MARKER_SIZE)

plt.xlabel('Time (Days)')
plt.ylabel('P(t)')
plt.legend(loc='upper right')
ax.set_xlim([0,275])
ax.set_ylim([0,8000])
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.set_axisbelow(True)
plt.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_mean_diagnosed.png')


# %%
# Confirmed Cases
dict_cases_scenarios_omega = {
    r'${t_{1/2}} = $' +  f'{half_life1} Days': C1_predicted[:, -1],
    r'${t_{1/2}} = $' +  f'{half_life2} Days': C2_predicted[:, -1],
    'Original': C_predicted[:, -1]
}
pd_cases_scenarios_omega = pd.DataFrame(dict_cases_scenarios_omega)

# Deaths
dict_deaths_scenarios_omega = {
    r'${t_{1/2}} = $' +  f'{half_life1} Days': D1_predicted[:, -1],
    r'${t_{1/2}} = $' +  f'{half_life2} Days': D2_predicted[:, -1],
    'Original': D_predicted[:, -1]
}
pd_deaths_scenarios_omega = pd.DataFrame(dict_deaths_scenarios_omega)


# %%
# *-------------------------------------------------*
# | Plot: Last Day - Frequency Plot Confirmed Cases |
# *-------------------------------------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
pd_cases_scenarios_omega.plot.hist(bins=60, alpha=0.8, ax = plt.gca(), color=['plum','mediumpurple',CCOLOR])
plt.xlabel('Confirmed Cases')

ax.set_ylim([0,450])
ax.set_axisbelow(True)
plt.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_confirmed.png')

# *----------------------------------------*
# | Plot: Last Day - Frequency Plot Deaths |
# *----------------------------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
pd_deaths_scenarios_omega.plot.hist(bins=60, alpha=0.8, ax = plt.gca(), color=['pink','lightcoral',DCOLOR])
plt.xlabel('Dead Cases')

ax.set_ylim([0,400])
#plt.xticks([32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000])
ax.set_axisbelow(True)
plt.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_dead.png')


# %%
style = dict(linestyle='-', linewidth=1.7)

# *--------------------------------------------*
# | Box Plot: Last Day - Frequency Plot Deaths |
# *--------------------------------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
pd_cases_scenarios_omega.plot.box(widths=0.4, ax = plt.gca(), boxprops=style, medianprops=style, color=dict(boxes='black', whiskers='black', medians='red', caps='black'))
plt.ylabel('Confirmed Cases')

ax.set_ylim([30000,130000])
ax.yaxis.set_major_locator(MultipleLocator(10000))
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_box_confirmed.png')

# *--------------------------------------------*
# | Box Plot: Last Day - Frequency Plot Deaths |
# *--------------------------------------------*
fig, ax = plt.subplots(figsize=(9,7), dpi=300)
pd_deaths_scenarios_omega.plot.box(widths=0.4, ax = plt.gca(), boxprops=style, medianprops=style, color=dict(boxes='black', whiskers='black', medians='red', caps='black'))
plt.ylabel('Dead Cases')

ax.set_ylim([4000,20000])
ax.yaxis.set_major_locator(MultipleLocator(2000))
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=ALPHA)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_prediction_box_dead.png')

