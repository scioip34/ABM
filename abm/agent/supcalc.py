"""
calc.py : Supplementary methods and calculations necessary for agents
"""
import numpy as np
from scipy import integrate
from abm.contrib import vswrm


def compute_state_variables(vel_now, Phi, V_now, t_now=None, V_prev=None, t_prev=None):
    """Calculating state variables of a given agent according to the main algorithm as in
    https://advances.sciencemag.org/content/6/6/eaay0792.
        Args:
            vel_now: current speed of the agent
            V_now: current binary visual projection field array
            Phi: linspace numpy array of visual field axis
            t_now: current time
            V_prev: previous binary visual projection field array
            t_prev: previous time
        Returns:
            dvel: temporal change in agent velocity
            dpsi: temporal change in agent heading angle

    """
    # # Deriving over t
    # if V_prev is not None and t_prev is not None and t_now is not None:
    #     dt = t_now - t_prev
    #     logger.debug('Movement calculation called with NONE as time-related parameters.')
    #     joined_V = np.vstack((V_prev, t_prev))
    #     dt_V = dt_V_of(dt, joined_V)
    # else:
    #     dt_V = np.zeros(len(Phi))

    dt_V = np.zeros(len(Phi))

    # Deriving over Phi
    dPhi_V = dPhi_V_of(Phi, V_now)

    # Calculating series expansion of functional G
    G_vel = (-V_now + vswrm.ALP2 * dt_V)

    # Spikey parts shall be handled separately because of numerical integration
    G_vel_spike = np.square(dPhi_V)

    G_psi = (-V_now + vswrm.BET2 * dt_V)

    # Spikey parts shall be handled separately because of numerical integration
    G_psi_spike = np.square(dPhi_V)

    # Calculating change in velocity and heading direction
    dPhi = Phi[-1] - Phi[-2]
    FOV_rescaling_cos = 1
    FOV_rescaling_sin = 1

    dvel = vswrm.GAM * (vswrm.V0 - vel_now) + \
           vswrm.ALP0 * integrate.trapz(np.cos(FOV_rescaling_cos * Phi) * G_vel, Phi) + \
           vswrm.ALP0 * vswrm.ALP1 * np.sum(np.cos(Phi) * G_vel_spike) * dPhi
    dpsi = vswrm.BET0 * integrate.trapz(np.sin(Phi) * G_psi, Phi) + \
           vswrm.BET0 * vswrm.BET1 * np.sum(np.sin(FOV_rescaling_sin * Phi) * G_psi_spike) * dPhi

    return dvel, dpsi


def dPhi_V_of(Phi, V):
    """Calculating derivative of VPF according to Phi visual angle array at a given timepoint t
        Args:
            Phi: linspace numpy array of visual field axis
            V: binary visual projection field array
        Returns:
            dPhi_V: derivative array of V w.r.t Phi
    """
    # circular padding for edge cases
    padV = np.pad(V, (1, 1), 'wrap')
    dPhi_V_raw = np.diff(padV)

    # we want to include non-zero value if it is on the edge
    if dPhi_V_raw[0] > 0 and dPhi_V_raw[-1] > 0:
        dPhi_V_raw = dPhi_V_raw[0:-1]

    else:
        dPhi_V_raw = dPhi_V_raw[1:, ...]

    dPhi_V = dPhi_V_raw / (Phi[-1] - Phi[-2])
    return dPhi_V


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if v1_u[0] * v2_u[1] - v1_u[1] * v2_u[0] < 0:
        angle = -angle
    return angle
