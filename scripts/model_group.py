from model import *

# SEAIRPD-Q Model
class GroupModel(Model):
	def __init__(self, start, stop, dt, params = None):
		super().__init__(start, stop, dt)
		
		if params is not None:
			self._define_parameters(*params)

	def _define_parameters(self, 
						   beta = 5.965935e-09,
						   mu = 5.965935e-09,
						   omega = 1.970400e-02, 
						   sigma = 1/5,
						   rho = 0.85, 
						   epsilon_i = 1/3,
						   gamma_a = 1/14, 
						   gamma_i = 1/14, 
						   gamma_p = 1/14,
						   di = 1.356770e-02, 
						   dp = 4.168171e-03, 
						   S0 = 210147006, 
						   E0 = 70, 
						   A0 = 7.0,
						   I0 = 35, 
						   P0 = 7.0, 
						   R0 = 0.0, 
						   D0 = 0.0,
						   eta = 0.0):

		
		# *------------------*
		# | Model Parameters |
		# *------------------*
		

		self.beta      = beta 			# Conversion rate parameter due to the interaction between susceptible and infected individuals
		self.mu        = mu			 	# Conversion rate parameter for susceptible individuals that interact with exposed individuals and then become exposed
		self.omega     = omega 			# Conversion rate parameter for susceptible, exposed and infected individuals that become recovered due to quarantine removal
		self.sigma     = sigma        	# Transition rate from exposed to infected class
		self.eta       = eta 				# Rate at which quarantined individuals become susceptible
		self.rho       = rho 			# Proportion of exposed individuals becoming symptomatic (thus the proportion of becoming asymptomatic is 1 - rho)
		self.epsilon_i = epsilon_i 		# Diagnostic rate of symptomatic infectious
		self.gamma_a   = gamma_a 		# Mean recovery period of class A
		self.gamma_i   = gamma_i 		# Mean recovery period of class I
		self.gamma_p   = gamma_p 		# Mean recovery period of class P
		self.di        = di 			# Disease-induced death rate
		self.dp        = dp 			# Disease-induced death rate

		# *--------------------*
		# | Initial Conditions |
		# *--------------------*
		
		self.S0 = S0
		self.E0 = E0
		self.A0 = A0
		self.I0 = I0
		self.P0 = P0
		self.R0 = R0
		self.D0 = D0

		self.IC = [self.S0, self.E0, self.A0, self.I0, self.P0, self.R0, self.D0]

	def _model(self, x, t):
		# *--------------------*
		# | Model Compartments |
		# *--------------------*

		S = x[0] # Susceptible
		E = x[1] # Exposed
		A = x[2] # Asymptomatically Infected
		I = x[3] # Symptomatically Infected
		P = x[4] # Positively Diagnosed
		R = x[5] # Recovered
		D = x[6] # Dead
		
		# *-----------------*
		# | Model Equations |
		# *-----------------*

		dSdt = -self.beta*S*I - self.mu*S*A - self.omega*S + self.eta*R
		dEdt = self.beta*S*I + self.mu*S*A - self.sigma*E - self.omega*E
		dAdt = self.sigma*(1 - self.rho)*E - self.gamma_a*A - self.omega*A
		dIdt = self.sigma*self.rho*E - self.gamma_i*I - self.di*I - self.omega*I - self.epsilon_i*I
		dPdt = self.epsilon_i*I - self.gamma_p*P - self.dp*P
		dRdt = self.gamma_a*A + self.gamma_i*I + self.gamma_p*P + self.omega*(S + E + A + I) - self.eta*R
		dDdt = self.di*I + self.dp*P

		return [dSdt, dEdt, dAdt, dIdt, dPdt, dRdt, dDdt]
	
	def calculate_R0(self):
		# Defined: Sandra Malta
		return ((self.beta*self.sigma*self.rho)/((self.sigma + self.omega)*(self.gamma_i + self.di + self.omega + self.epsilon_i)) + (self.sigma*(1 - self.rho)*self.mu)/((self.sigma + self.omega)*(self.gamma_a + self.omega)))*self.S0
