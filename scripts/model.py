import numpy as np
from scipy.integrate import odeint
from abc import ABC, abstractmethod

class Model(ABC):
	def __init__(self, start, stop, dt):
		self.dt = dt
		self.start = start
		self.stop = stop
		self.IC = None
		self._define_parameters()

	@abstractmethod
	def _define_parameters(self):
		pass

	@abstractmethod
	def _model(self):
		pass

	@abstractmethod
	def calculate_R0(self):
		pass

	def _define_time_space(self, start, stop, dt):
		return np.arange(start, stop + dt, dt)

	def solve_ode(self):
		self.t = self._define_time_space(self.start, self.stop, self.dt)
		self.x = odeint(self._model, self.IC, self.t)
		return self.x, self.t
