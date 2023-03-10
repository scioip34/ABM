import numpy as np

from abm.agent.supcalc import angle_between, find_nearest

from scipy import integrate

def distance_infinite(p1, p2, L=500, dim=2):
    """ Returns the distance vector of two position vectors x,y
        by tanking periodic boundary conditions into account.
        Input parameters: L - system size, dim - no. of dimension
    """
    distvec = p2 - p1
    distvec_periodic = np.copy(distvec)
    distvec_periodic[distvec < -0.5*L] += L
    distvec_periodic[distvec > 0.5*L] -= L
    return distvec_periodic

def projection_field(fov, v_field_resolution, position, radius,
                     orientation, object_positions,
                     boundary_cond="walls", arena_width=None, arena_height=None, vision_range=None, ag_id=0):
    """
    Calculating visual projection field for the agent given the visible
    obstacles in the environment
    obstacle sprites to generate projection field
    :param fov: tuple of number with borders of fov such as (-np.pi, np.pi)
    :param v_field_resolution: visual field resolution in pixels
    :param position: np.xarray of agent's position
    :param radius: radius of the agent
    :param orientation: orientation angle between 0 and 2pi
    :param max_proj_size: maximum projection size to include in the visual proj. field
    :param boundary_cond: boundary condition either infinite or walls. If walls projection field
    is calculated according to euclidean coordinates in 2D space, otherwise on a torus.
    :return: projection field np.xarray with shape (n objects, field resolution)
    """
    # initializing visual field and relative angles
    v_field = np.zeros((len(object_positions), v_field_resolution))
    phis = np.linspace(-np.pi, np.pi, v_field_resolution)

    # center point
    agents_center = position + radius

    # point on agent's edge circle according to it's orientation
    agent_edge = position + np.array(
        [1 + np.cos(orientation), 1 - np.sin(orientation)]) * radius

    # vector between center and edge according to orientation
    v1 = agent_edge - agents_center

    # 1. Calculating closed angle between object and agent according to the
    # position of the object.
    # 2. Calculating visual projection size according to visual angle on
    # the agent's retina according to distance between agent and object
    for i, obj_position in enumerate(object_positions):
        # continue if the object and the agent position completely coincide
        if obj_position[0] == position[0] and obj_position[1] == position[1]:
            continue

        # center of obstacle (as it must be another agent)
        object_center = obj_position + radius

        # vector between agent center and object center
        v2 = object_center - agents_center

        # in case torus, positions might change
        if boundary_cond == "infinite":
            if np.abs(v2[0]) > arena_width/2:
                if agents_center[0] < object_center[0]:
                    object_center[0] -= arena_width
                elif agents_center[0] > object_center[0]:
                    object_center[0] += arena_width
            if np.abs(v2[1]) > arena_height/2:
                if agents_center[1] < object_center[1]:
                    object_center[1] -= arena_height
                elif agents_center[1] > object_center[1]:
                    object_center[1] += arena_height

            # recalculating v2 after teleporting on torus
            v2 = object_center - agents_center

        # calculating closed angle between v1 and v2
        closed_angle = calculate_closed_angle(v1, v2)

        distance = np.linalg.norm(object_center - agents_center)

        # limiting vision range if requested
        if vision_range is not None:
            if distance > vision_range:
                continue

        # calculating the visual angle from focal agent to target
        vis_angle = 2 * np.arctan(radius / (1 * distance))

        # finding where in the retina the projection belongs to
        phi_target = find_nearest(phis, closed_angle)

        # transforming fov boundaries to pixel boundaries
        fov_px = [find_nearest(phis, fov[0]), find_nearest(phis, fov[1])]

        # # if target is visible we save its projection into the VPF
        # # source data
        # if fov[0] <= closed_angle <= fov[1]:
        # the projection size is proportional to the visual angle.
        # If the projection is maximal (i.e. taking each pixel of the
        # retina) the angle is 2pi from this we just calculate the
        # projection size using a single proportion
        proj_size = (vis_angle / (2 * np.pi)) * v_field_resolution

        proj_start = int(phi_target - np.floor(proj_size / 2))
        proj_end = int(phi_target + np.floor(proj_size / 2))

        if fov_px[0] < proj_start < fov_px[1] or fov_px[0] < proj_end < fov_px[1]:

            # circular boundaries to the VPF as there is 360 degree vision
            if proj_start < 0:
                v_field[i, v_field_resolution + proj_start:v_field_resolution] = 1
                proj_start = 0
            if proj_end >= v_field_resolution:
                v_field[i, 0:proj_end - (v_field_resolution - 1)] = 1
                proj_end = v_field_resolution

            v_field[i, proj_start:proj_end] = 1


    # post_processing and limiting FOV
    # flip field data along second dimension
    v_field_post = np.flip(v_field, axis=1)

    # v_field_post[:, phis < fov[0]] = 0
    # v_field_post[:, phis > fov[1]] = 0
    return v_field_post



def calculate_closed_angle(v1, v2):
    """
    Calculating closed angle between two vectors v1 and v2. Rotated with the orientation of the agent as it is relative.
    :param v1: vector 1; np.xarray
    :param v2: vector 2; np.xarray
    :return: closed angle between v1 and v2
    """
    closed_angle = angle_between(v1, v2)
    closed_angle = (closed_angle % (2 * np.pi))
    # at this point closed angle between 0 and 2pi, but we need it between -pi and pi
    # we also need to take our orientation convention into consideration to
    # recalculate theta=0 is pointing to the right
    if 0 <= closed_angle <= np.pi:
        closed_angle = -closed_angle
    else:
        closed_angle = 2 * np.pi - closed_angle
    return closed_angle

# Functions needed for VSWRM functionality
def VSWRM_flocking_state_variables(vel_now, Phi, V_now, vf_params, t_now=None, V_prev=None, t_prev=None):
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
    G_vel = (-V_now + vf_params.ALP2 * dt_V)

    # Spikey parts shall be handled separately because of numerical integration
    G_vel_spike = np.square(dPhi_V)

    G_psi = (-V_now + vf_params.BET2 * dt_V)

    # Spikey parts shall be handled separately because of numerical integration
    G_psi_spike = np.square(dPhi_V)

    # Calculating change in velocity and heading direction
    dPhi = Phi[-1] - Phi[-2]
    FOV_rescaling_cos = 1
    FOV_rescaling_sin = 1

    # print(f"alp0 : {vswrm.ALP0 * integrate.trapz(np.cos(FOV_rescaling_cos * Phi) * G_vel, Phi)}", )
    # print(f'alp1 : {vswrm.ALP0 * vswrm.ALP1 * np.sum(np.cos(Phi) * G_vel_spike) * dPhi}')

    # dvel = vf_params.GAM * (vf_params.V0 - vel_now) + \
    #        vf_params.ALP0 * integrate.trapz(np.cos(FOV_rescaling_cos * Phi) * G_vel, Phi) + \
    #        vf_params.ALP0 * vf_params.ALP1 * np.sum(np.cos(Phi) * G_vel_spike) * dPhi
    # dpsi = vf_params.BET0 * integrate.trapz(np.sin(Phi) * G_psi, Phi) + \
    #        vf_params.BET0 * vf_params.BET1 * np.sum(np.sin(FOV_rescaling_sin * Phi) * G_psi_spike) * dPhi

    # without reacling edge information
    dvel = vf_params.GAM * (vf_params.V0 - vel_now) + \
           vf_params.ALP0 * integrate.trapz(np.cos(Phi) * G_vel, Phi) + \
           vf_params.ALP0 * vf_params.ALP1 * np.sum(np.cos(Phi) * G_vel_spike)

    dpsi = vf_params.BET0 * integrate.trapz(np.sin(Phi) * G_psi, Phi) + \
           vf_params.BET0 * vf_params.BET1 * np.sum(np.sin(Phi) * G_psi_spike)

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

    dPhi_V = dPhi_V_raw #/ (Phi[-1] - Phi[-2])
    return dPhi_V