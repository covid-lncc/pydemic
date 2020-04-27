# Plotting libs
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
from scipy.integrate import solve_ivp  # to solve ODE system
from tqdm import tqdm


def seirpdq_model(
    t,
    X,
    beta=1e-7,
    mu=1e-7,
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
    SEAIRPD-Q python implementation.
    """
    S, E, A, I, P, R, D = X  # unpacking state variables from input

    S_prime = -beta / N * S * I - mu / N * S * A - omega * S + eta * R
    E_prime = beta / N * S * I + mu / N * S * A - sigma * E - omega * E
    A_prime = sigma * (1 - rho) * E - gamma_A * A - omega * A
    I_prime = sigma * rho * E - gamma_I * I - d_I * I - omega * I - epsilon_I * I
    P_prime = epsilon_I * I - gamma_P * P - d_P * P
    R_prime = gamma_A * A + gamma_I * I + gamma_P * P + omega * (S + E + A + I) - eta * R
    D_prime = d_I * I + d_P * P
    return S_prime, E_prime, A_prime, I_prime, P_prime, R_prime, D_prime


def seairpdq_ode_solver(
    y0,
    t_eval,
    t_span=None,
    beta=1e-7,
    mu=1e-7,
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
    N=1,
    method="Radau",
):
    """
    SciPy ODE solver wrapper.
    """
    if t_span is None:
        t_span = (t_eval.min(), t_eval.max())

    solution_ODE = solve_ivp(
        fun=lambda t, y: seirpdq_model(
            t,
            y,
            beta=beta,
            mu=mu,
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
        method=method,
    )

    return solution_ODE


def run_bifurcation_analysis(
    ode_solver_wrapper,
    bifurcation_variable_name,
    parameter_span,
    parameter_num_of_points,
    fixed_parameters_dict,
    time_span,
    time_num_of_points,
    initial_conditions,
    output_variable_name_list,
    num_of_last_outcomes,
):
    if type(bifurcation_variable_name) != str:
        raise ValueError("Input bifurcation_variable_name must be of str type.")

    if type(parameter_num_of_points) != int:
        raise ValueError("Input parameter_num_of_points must be of integer type.")

    if type(time_num_of_points) != int:
        raise ValueError("Input time_num_of_points must be of integer type.")

    parameter_span_type = type(parameter_span)
    if parameter_span_type != tuple and parameter_span_type != list:
        raise ValueError("Input parameter_span must be a list or tuple of values.")
    else:
        if len(parameter_span) != 2:
            raise ValueError("Input parameter_span must be a list or tuple with two values.")

    if len(initial_conditions) != len(output_variable_name_list):
        raise ValueError(
            "The number of output variables from ODE model must be equal to number of initial conditions."
        )

    parameter_values = np.linspace(parameter_span[0], parameter_span[1], parameter_num_of_points)
    bifurcation_parameter_values_dict = {bifurcation_variable_name: parameter_values}
    time_values = np.linspace(time_span[0], time_span[1], time_num_of_points)

    # Creating dictionary to record bifurcation realizations
    output_variables_dict = dict()
    for output_variable_name in output_variable_name_list:
        output_variables_dict[output_variable_name] = list()

    # Running bifurcation evaluations
    parameter_progress_bar = tqdm(parameter_values)
    for parameter_value in parameter_progress_bar:
        parameter_progress_bar.set_description("Running continuation")
        parameters = fixed_parameters
        parameters[bifurcation_variable_name] = parameter_value
        ode_solver_output = ode_solver_wrapper(initial_conditions, time_values, **parameters)
        t_output, y_output = ode_solver_output.t, ode_solver_output.y
        y_last_outputs = y_output[:, -num_of_last_outcomes:]
        output_index = 0
        for output_variable_name in output_variable_name_list:
            output_variable_last_results = y_last_outputs[output_index]
            output_variables_dict[output_variable_name].append(
                [output_variable_last_results.min(), output_variable_last_results.max()]
            )
            output_index += 1

    return bifurcation_parameter_values_dict, output_variables_dict


fixed_parameters = {
    "omega": 4.03372912e-02,
    "beta": 7.15533505e-09,
    "mu": 3.91298105e-13,
    "sigma": 1.53844449e-01,
    "rho": 8.99815662e-01,
    "epsilon_I": 6.61939072e-02,
    "gamma_I": 5.26431146e-02,
    "gamma_P": 5.26329128e-02,
    "gamma_A": 6.94329867e-02,
    "d_I": 1.00039701e-04,
    "d_P": 6.32791443e-03,
    # "eta": 0
}

eta_num_of_points = 50
eta_min = 0
eta_max = 0.02
eta_points = np.linspace(eta_min, eta_max, eta_num_of_points)
parameter_to_bifurcation = {"eta": eta_points}

seairpdq_y0 = [
    210147009,
    64,
    1,
    253,
    13,
    0,
    0,
]

eta_values, y_last_results = run_bifurcation_analysis(
    ode_solver_wrapper=seairpdq_ode_solver,
    bifurcation_variable_name="eta",
    parameter_span=(0, 0.02),
    parameter_num_of_points=100,
    fixed_parameters_dict=fixed_parameters,
    time_span=(0, 50000),
    time_num_of_points=50000,
    initial_conditions=seairpdq_y0,
    output_variable_name_list=["S", "E", "A", "I", "P", "R", "D"],
    num_of_last_outcomes=1000,
)

colors_list = ["b", "g", "r", "c", "m", "y", "k"]
results_progress_bar = tqdm(y_last_results)
num_of_state_variables = len(y_last_results)
fig, axs = plt.subplots(num_of_state_variables, sharex=True, gridspec_kw={"hspace": 1})
index = 0
for variable_name in results_progress_bar:
    results_progress_bar.set_description("Plotting results")
    variable_result = np.array(y_last_results[variable_name])
    variable_min_values = variable_result[:, 0]
    variable_max_values = variable_result[:, 1]

    color = colors_list[index]
    axs[index].fill_between(
        eta_values["eta"], variable_min_values, variable_max_values, color=color, alpha=1
    )

    index += 1

for ax, variable_name in zip(axs.flat, y_last_results):
    ax.set(xlabel=r"$\eta$", ylabel=f"{variable_name}", xlim=(eta_min, eta_max))
    ax.label_outer()

plt.savefig("eta_continuation.png", dpi=300)
