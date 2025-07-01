import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import time

@dataclass
class Params:
    # Time
    dt: float = 0.1
    t0: float = 0
    tf: float = 50

    # Constants
    m: float = 1000 # kg
    g: float = 9.81 # m/s^2
    radius: float = 0.3 # m
    b: float = 0.01 # kg*m^2 / s
    J: float = 0.5 # kg*m^2
    rho: float = 1.293 #kg*m^-3
    Af: float = 2.5 # m^2
    Cd: float = 0.25
    Crr: float = 0.01
    mu : float = 0.7
    etta: float = 4 # drivetrain from motor to wheels
    T_min: float = -200
    theta: float = 0

    # Pacejka Params
    B: float = 10
    C: float = 2
    E: float = 0.8

    # Tunable Params
    v0: float = 20 # m/s
    epsilon: float = 0.02
    Kp: float = 300 #300
    Ki: float = 20 #20
    Kd: float = 1 #1
    Kff: float = 0.7 #0.7

class ABS:
    
    def __init__(self, params = Params()):
        # imported constants
        self.dt = params.dt
        self.t0 = params.t0
        self.tf = params.tf

        self.m = params.m
        self.g = params.g
        self.radius = params.radius
        self.b = params.b
        self.J = params.J
        self.rho = params.rho
        self.Af = params.Af
        self.Cd = params.Cd
        self.Crr = params.Crr
        self.mu = params.mu
        self.etta = params.etta
        self.T_min = params.T_min
        self.theta = params.theta

        self.B = params.B
        self.C = params.C
        self.E = params.E

        self.epsilon = params.epsilon
        self.Kp = params.Kp
        self.Ki = params.Ki
        self.Kd = params.Kd
        self.Kff = params.Kff

        # stored values
        self.error_prev = 0
        self.error_sum = 0
        self.sigma_prev = 0

        self.T_motor = 0

        v0 = params.v0
        omega0 = v0/self.radius
        self.init = np.array([v0, omega0])

    # Forces
    def F_aero(self, v):
        return 0.5 * self.rho * self.Cd * self.Af * v**2
    
    def Fz(self, theta):
        return self.m * self.g * np.cos(theta)
    
    def Frr(self, v_wheel, theta):
        k = 5
        Fz = self.Fz(theta)
        return self.Crr * Fz * np.tanh(k * v_wheel)
    
    def sigma(self, v, v_wheel):
        if v > 0:
            return (v_wheel - v)/v
        else:
            return 0
        
    def Fx(self, sigma, theta):
        D = self.mu * self.Fz(theta)
        Fx = D * np.sin(self.C * np.arctan(self.B * sigma - self.E * (self.B * sigma - np.arctan(self.B * sigma))))
        return Fx
    
    # Torques
    def Tf(self, omega):
        return self.b * omega
    
    def T_brake_abs(self, v, v_wheel, v_wheel_ref, sigma):
        if v > 0 and v_wheel > 0:
            T_pid = self.PID(v_wheel, v_wheel_ref)
            T_ff = self.Kff * (sigma - self.sigma_prev)/self.dt
            T_total = T_pid + T_ff
            return max(self.T_min, T_total)
        else:
            return 0
    
    def T_brake_input(self, v, v_wheel, v_wheel_ref):
        if v > 0 and v_wheel > 0:
            return self.T_min
        else:
            return 0
    
    def T_rr(self, v_wheel, theta):
        return self.Frr(v_wheel, theta) * self.radius
    
    # Dynamics
    def dynamics_abs(self, t, states):
        v = states[0]
        omega = states[1]

        v_wheel = omega*self.radius
        sigma = self.sigma(v, v_wheel)
        v_wheel_ref = self.v_wheel_ref(v, self.theta)

        F_aero = self.F_aero(v)
        Fz = self.Fz(self.theta)
        Frr = self.Frr(v_wheel, self.theta)
        Fx = self.Fx(sigma, self.theta)

        Tf = self.Tf(omega)
        T_rr = self.T_rr(v_wheel, self.theta)
        T_brake = self.T_brake_abs(v, v_wheel, v_wheel_ref, sigma)

        F_long = -F_aero - Frr + Fx
        T_net = -Tf + T_brake - T_rr

        return np.array([F_long/self.m, T_net/self.J])
    
    def dynamics(self, t, states):
        v = states[0]
        omega = states[1]
        # self.theta = np.random.uniform(-np.pi/10, np.pi/10)

        v_wheel = omega*self.radius
        sigma = self.sigma(v, v_wheel)
        v_wheel_ref = self.v_wheel_ref(v, self.theta)

        F_aero = self.F_aero(v)
        Fz = self.Fz(self.theta)
        Frr = self.Frr(v_wheel, self.theta)
        Fx = self.Fx(sigma, self.theta)

        Tf = self.Tf(omega)
        T_rr = self.T_rr(v_wheel, self.theta)
        T_brake = self.T_brake_input(v, v_wheel, v_wheel_ref)

        F_long = -F_aero - Frr + Fx
        T_net = -Tf + T_brake - T_rr

        return np.array([F_long/self.m, T_net/self.J])

    # Helper Functions
    def sigma_min(self, theta):
        sigma = np.linspace(-1, 1, 1000)
        D = self.mu * self.Fz(theta)

        Fx = D * np.sin(self.C * np.arctan(self.B * sigma - self.E * (self.B * sigma - np.arctan(self.B * sigma))))
        sigma_min = sigma[np.argmin(Fx)]

        return sigma_min

    def v_wheel_ref(self, v, theta):
        sigma_des = self.sigma_min(theta)
        return v*(1 + sigma_des)

    def PID(self, v_wheel, v_wheel_ref):
        error = v_wheel_ref - v_wheel
        u = self.Kp*error +self.Ki*self.error_sum*self.dt + self.Kd*(error - self.error_prev)/self.dt
        self.error_prev = error
        self.error_sum += error
        return u
    
    # Main Function
    def step(self, abs):
    
        t_eval = np.linspace(self.t0, self.tf, 1000)

        def stop_event(t, y):
            return y[0]
        stop_event.terminal = True
        stop_event.direction = -1

        if abs == 'ON':
            sol = solve_ivp(self.dynamics_abs, [self.t0, self.tf], self.init, t_eval = t_eval, events = stop_event)
            t = sol.t
            y = sol.y

            return(t, y)
        elif abs == 'OFF':
            sol = solve_ivp(self.dynamics, [self.t0, self.tf], self.init, t_eval = t_eval, events = stop_event)
            t = sol.t
            y = sol.y

            return(t, y)
    
if __name__ == "__main__":
    params = Params()
    abs = ABS(params)

    t_abs, y_abs = abs.step('ON')
    t_off, y_off = abs.step('OFF')
    t_max = max(t_abs[-1], t_off[-1])

    v_abs = y_abs[0]
    v_wheel_abs = y_abs[1] * abs.radius

    v_off = y_off[0]
    v_wheel_off = y_off[1] * abs.radius

    sigma_abs = (v_wheel_abs - v_abs)/v_abs
    sigma_off = (v_wheel_off - v_off)/v_off

    sigma_abs = np.clip(sigma_abs, -1.0, 1.0)
    sigma_off = np.clip(sigma_off, -1.0, 1.0)


    x_abs = np.trapz(y_abs[0], t_abs)
    x_off = np.trapz(y_off[0], t_off)

    sigma = np.linspace(-1, 1, 100)
    Fx = abs.Fx(sigma, 0)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t_abs, v_abs, label='car speed')
    ax[0].plot(t_abs, v_wheel_abs, label='wheel speed')
    ax[0].set_title(f'Stopping distance (ABS ON):  {x_abs:.2f} m')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Speed (m/s)')
    ax[0].set_xlim([0, t_max])
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t_off, v_off, label='car speed')
    ax[1].plot(t_off, v_wheel_off, label='wheel speed')
    ax[1].set_title(f'Stopping distance (ABS OFF): {x_off:.2f} m')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Speed (m/s)')
    ax[1].grid()
    ax[1].set_xlim([0, t_max])
    ax[1].legend()

    plt.tight_layout(pad=2.0)

    fig_p, ax_p = plt.subplots()
    plt.ion()
    ax_p.plot(sigma, Fx, color='black')
    ax_p.set_title('Traction Animation')
    ax_p.set_xlabel('Slip Ratio')
    ax_p.set_ylabel('Traction Force')
    ax_p.grid()
    point_off = ax_p.scatter(0, 0, color='red', s=50)
    point_abs = ax_p.scatter(0, 0, color='green', s=50)  

    for k in range(max(len(v_off), len(v_abs))):
        if k < len(v_off) - 1 and k < len(sigma_off) - 1 and k < len(t_off) - 1:
            if v_off[k] > 0:
                Fx_off = abs.Fx(sigma_off[k], 0)
                point_off.remove()
                point_off = ax_p.scatter(sigma_off[k], Fx_off, color='red', s=50, label=f'ABS Off\ntime: {t_off[k]:.2f}s')

        if k < len(v_abs) - 1 and k < len(sigma_abs) - 1 and k < len(t_abs) - 1:
            if v_abs[k] > 0:
                Fx_abs = abs.Fx(sigma_abs[k], 0)
                point_abs.remove()
                point_abs = ax_p.scatter(sigma_abs[k], Fx_abs, color='green', s=50, label=f'ABS On\ntime: {t_abs[k]:.2f}s')
        ax_p.legend()
        plt.pause(0.001)

    plt.ioff()
    plt.show()