from pydantic import BaseSettings, BaseModel


class DecisionParameters(BaseModel):
    """Decision variables"""
    # W
    T_w: float
    # Eps_w: float = 3
    # g_w: float = 0.085
    # B_w: float = 0
    # w_max: float = 1
    #
    # # U
    # T_u: float = 0.5
    # Eps_u: float = 3
    # g_u: float = 0.085
    # B_u: float = 0
    # u_max: float = 1
    #
    # # Inhibition
    # S_wu: float = 0.25
    # S_uw: float = 0.01
    #
    # # Calculating Private Information
    # Tau: int = 10
    # F_N: float = 2
    # F_R: float = 1


class MovementParameters(BaseModel):
    """Movement variables"""

    # Exploration movement parameters
    exp_vel_min: float = 1
    exp_vel_max: float = 1
    exp_theta_min: float = -0.3
    exp_theta_max: float = 0.3

    # Relocation movement parameters
    reloc_des_vel: float = 1
    reloc_theta_max: float = 0.5

    # Exploitation params
    # deceleration when a patch is reached
    exp_stop_ratio: float = 0.08


class AgentParameters(BaseSettings):
    test: str
    # movement_parameters: MovementParameters
    # dec: DecisionParameters

    class Config:
        env_nested_delimiter = '__'

