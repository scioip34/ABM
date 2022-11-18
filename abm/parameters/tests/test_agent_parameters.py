from pathlib import Path

from abm.parameters.agent_parameters import AgentParameters


def test_agent_parameters():
    """
    Testing agent parameters
    SEE: https://pydantic-docs.helpmanual.io/usage/settings/#dotenv-env-support
    """
    test_env_file = Path(__file__).parent / 'data' / 'test.env'

    # NOTE: _env_file overrides the default Config in the class
    params = AgentParameters(_env_file=test_env_file)

    # assert params.dec.T_w == 0.2
    assert params.test == 't'
