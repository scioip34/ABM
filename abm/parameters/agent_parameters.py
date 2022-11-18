from pydantic import BaseSettings, BaseModel


class DecisionParameters(BaseModel):
    """Decision variables"""
    # NOTE: error when trying to parse an uppercase variables from .env
    # SEE: https://github.com/pydantic/pydantic/issues/3936#issuecomment-1152903692
    # W
    t_w: float = 0.5
    eps_w: float = 3
    g_w: float = 0.085
    b_w: float = 0
    w_max: float = 1

    # U
    t_u: float = 0.5
    eps_u: float = 3
    g_u: float = 0.085
    b_u: float = 0
    u_max: float = 1

    # Inhibition
    s_wu: float = 0.25
    s_uw: float = 0.01

    # Calculating Private Information
    tau: int = 10
    f_n: float = 2
    f_r: float = 1


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
    agent_movement: MovementParameters
    agent_decision: DecisionParameters

    class Config:
        env_nested_delimiter = '__'

