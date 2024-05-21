class ForwardEuler:
    def __init__(self, rhs, dt):
        """
        Initializes the Forward Euler method for a given right-hand side function
        and time step.

        Parameters
        ----------
        rhs : callable
            The right-hand side function of the ODE.
        dt : float
            The time step of the integration.
        """
        self.rhs = rhs
        self.dt = dt

    def step(self, psi, t0, direct=None):
        """
        Integrates the ODE
            dot(psi) = rhs(psi, t)
        from t0 to t0 + dt using the Forward Euler method.
        """
        if direct is None:
            return psi + self.dt * self.rhs(psi, t0)
        else:
            return psi + self.dt * self.rhs(psi, t0, direct)


class RungeKutta2:
    def __init__(self, rhs, dt):
        """
        Initializes the Runge-Kutta 2 method for a given right-hand side function
        and time step.

        Parameters
        ----------
        rhs : callable
            The right-hand side function of the ODE.
        dt : float
            The time step of the integration.
        """
        self.rhs = rhs
        self.dt = dt

    def step(self, psi, t0):
        """
        Integrates the ODE
            dot(psi) = rhs(psi, t)
        from t0 to t0 + dt using the Runge-Kutta 2 method.
        """
        k1 = self.rhs(psi, t0)
        k2 = self.rhs(psi + self.dt * k1, t0 + self.dt)
        return psi + self.dt / 2 * (k1 + k2)


class RungeKutta3:
    def __init__(self, rhs, dt):
        """
        Initializes the Runge-Kutta 3 method for a given right-hand side function
        and time step.

        Parameters
        ----------
        rhs : callable
            The right-hand side function of the ODE.
        dt : float
            The time step of the integration.
        """
        self.rhs = rhs
        self.dt = dt

    def step(self, psi, t0):
        """
        Integrates the ODE
            dot(psi) = rhs(psi, t)
        from t0 to t0 + dt using the Runge-Kutta 3 method.
        """
        k1 = self.rhs(psi, t0)
        k2 = self.rhs(psi + 0.5 * self.dt * k1, t0 + 0.5 * self.dt)
        k3 = self.rhs(psi - self.dt * k1 + 2 * self.dt * k2, t0 + self.dt)
        return psi + self.dt / 6 * (k1 + 4 * k2 + k3)


class RungeKutta4:
    def __init__(self, rhs, dt):
        """
        Initializes the Runge-Kutta 4 method for a given right-hand side function
        and time step.

        Parameters
        ----------
        rhs : callable
            The right-hand side function of the ODE.
        dt : float
            The time step of the integration.
        """
        self.rhs = rhs
        self.dt = dt

    def step(self, psi, t0, direct=None):
        """
        Integrates the ODE
            dot(psi) = rhs(psi, t)
        from t0 to t0 + dt using the Runge-Kutta 4 method.
        """
        if direct is None:
            k1 = self.rhs(psi, t0)
            k2 = self.rhs(psi + 0.5 * self.dt * k1, t0 + 0.5 * self.dt)
            k3 = self.rhs(psi + 0.5 * self.dt * k2, t0 + 0.5 * self.dt)
            k4 = self.rhs(psi + self.dt * k3, t0 + self.dt)
        else:
            k1 = self.rhs(psi, t0, direct)
            k2 = self.rhs(psi + 0.5 * self.dt * k1, t0 + 0.5 * self.dt, direct)
            k3 = self.rhs(psi + 0.5 * self.dt * k2, t0 + 0.5 * self.dt, direct)
            k4 = self.rhs(psi + self.dt * k3, t0 + self.dt, direct)
        return psi + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
