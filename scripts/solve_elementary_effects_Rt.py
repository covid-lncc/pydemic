# Sensitivity Analysis

import numpy as np
import scipy as sp
import SALib.sample.morris 
import SALib.analyze.morris
import matplotlib.pyplot as plt
import tikzplotlib as tikzplt
from model_group import *
from model_chinese import *


# *------------------*
# | Initial Settings |
# *------------------*

# Receives system solution at some time 
# returns the associated QoI of interest

def seiaprdq_get_qoi_ic(x, params):
    s, e, a, i, p, r, d = x
    beta, mu, omega, sigma, rho, epsilon_i, gamma_a, gamma_i, gamma_p, di, dp = params
    
    rt = s * ( ( beta * sigma * rho )/( (sigma + omega) * (gamma_i + di + omega + epsilon_i) )
        + 
        (sigma * (1 - rho) * mu) / ( (sigma + omega) * (gamma_a + omega)  ) 
       ) 
    
    return rt





# Seed for sampling
r_seed = 24
np.random.seed(r_seed)


# Problem parameters definition
# Rio 
# lsq_params = [4.658646e-08,
#              4.658646e-08,
#              1.438746e-02,
#              1/5,
#              0.85,
#              1/3,
#              1/14,
#              1/14,
#              1/14,
#              5.549912e-04,
#              1.315661e-02,
#              80,
#              8,
#              4]

# fixed_params = [ 17264837 ]

# Brasil
lsq_params = [5.965935e-09,
              5.965935e-09,
              1.970400e-02,
              1/5,
              0.85,
              1/3,
              1/14,
              1/14,
              1/14,
              1.356770e-02,
              4.168171e-03,
              70,
              7,
              35]


fixed_params = [210147006]




seiaprdq_problem = {
    'num_vars': 14,
    'names': ['beta', 'mu', 'omega', 'sigma', 'rho', 'epsilon_i', 'gamma_a', 'gamma_i', 'gamma_p', 'di', 'dp','e0', 'a0', 'i0'],
    'bounds': [ [param - 0.5 * param, param + 0.5 * param ] for param in lsq_params]
}
seiaprdq_problem["bounds"][4][1] = np.clip(seiaprdq_problem["bounds"][4][1], 0, 1)  # Fix for rho


# Generating parameter samples
p = 4    # Number of grid levels
d = 40   # Number of optimal trajectories
seiaprdq_param_values = SALib.sample.morris.sample(seiaprdq_problem, d, num_levels = p, local_optimization=False)


# *-------------*
# | Group Model SA (EEffects) |
# *-------------*

# Time Space
dt = 0.1
ini = 0.0
endtimes = np.arange(0, 260, 25)
endtimes[0] = 2.5


# QoIs
seiaprdq_qoi_values = np.zeros( (endtimes.shape[0], seiaprdq_param_values.shape[0]) )
# Sensitivity meters
seiaprdq_si = np.zeros(  (endtimes.shape[0], seiaprdq_problem["num_vars"]) )
# Solutions to be plotted
seiaprdq_solutions_sample = []

# Loop through all parametric space
for i in range(0, len(endtimes)):
    end = endtimes[i]
    for j, param in enumerate(seiaprdq_param_values):
        param_, ic_ = param[:11], param[11:] 
        param_full = np.concatenate( ( param_, fixed_params, ic_ ) )
        group_model = GroupModel(ini, end, dt, param_full)
        x, t = group_model.solve_ode()
        seiaprdq_qoi_values[i, j] = seiaprdq_get_qoi_ic( x[-1, :],  param_ )

        # Saving some solutions
        if( len(seiaprdq_solutions_sample) < 10 and np.random.rand() < 0.35 and i==len(endtimes) - 1 ):
            seiaprdq_solutions_sample.append([t, x, param_])    


    # Performing SA using Elementary effects method
    seiaprdq_si_i = SALib.analyze.morris.analyze(seiaprdq_problem, seiaprdq_param_values, seiaprdq_qoi_values[i, :], num_levels = p)
    # Normalizing metrics
    seiaprdq_si_i["mu_star"] = seiaprdq_si_i["mu_star"] / np.sum(seiaprdq_si_i["mu_star"]) 
    seiaprdq_si_i["sigma"] = seiaprdq_si_i["sigma"] / np.sum(seiaprdq_si_i["sigma"])
    
    # Saving results
    seiaprdq_si[i, :]  = seiaprdq_si_i["mu_star"]



# *------------------*
# | Outputs |
# *------------------*

# Plot definitions (SA, bars)
labels = endtimes # labels for each bar
width = 10     # the width of the bars
fig, ax = plt.subplots()
base_bar = np.zeros( endtimes.shape[0] )


for j in range( 0,  seiaprdq_problem["num_vars"]):
    if(j==0):
        ax.bar(labels, seiaprdq_si[:, j] , width, color = np.random.rand(3,), label=seiaprdq_problem["names"][j])
    else:
        ax.bar(labels, seiaprdq_si[:, j] , width, bottom = base_bar, color = np.random.rand(3,), label=seiaprdq_problem["names"][j])
        
    base_bar+= seiaprdq_si[:, j]
    print(seiaprdq_problem["names"][j], seiaprdq_si[:, j])
    print('\n')

ax.set_ylabel('First order sensitivity coefficient')
ax.legend()
ax.set_ylim(0, 1.15)

# Save output in tikz format
tikzplt.save("python_outputs/EE.tex")
plt.show(block = False)

#------------------------
# Model Results (samples, may be removed from code in the future)
for idx in range(0, len(seiaprdq_solutions_sample)):
    t, x, param = seiaprdq_solutions_sample[idx]

    # Saving parameters 
    output_file = open("python_outputs/param_{0}.txt".format(idx + 2), 'w')
    output_file.write( str(param) )
    output_file.close()

    # Plot
    plt.figure(idx + 2)
    i, p, r,  d = x[:, 3], x[:, 4], x[:,5], x[:, 6] 
    plt.plot(t, p, label = 'Diagnosed' + str(idx))
    plt.plot(t, i, label = 'Infected' + str(idx))
    plt.plot(t, d, label = 'Deaths' + str(idx))
    plt.legend()
    plt.show(block=False)
    

#------------------------
# Model Calibrated "mean" result
end = endtimes[-1]
param = lsq_params
group_model = GroupModel(ini, end, dt, param)
x, t = group_model.solve_ode()

# Saving parameters 
output_file = open("python_outputs/param_MAP.txt", 'w')
output_file.write( str(param) )
output_file.close()
    
# Plot
plt.figure(13)
i, p, r,  d = x[:, 3], x[:, 4], x[:,5], x[:, 6] 
plt.plot(t, p, label = 'Diagnosed' + '13')
plt.plot(t, i, label = 'Infected' + '13')
plt.plot(t, d, label = 'Deaths' + '13')
plt.legend()
plt.show()



