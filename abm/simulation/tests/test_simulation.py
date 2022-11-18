from pathlib import Path

from abm.parameters.playground_parameters import PlaygroundParameters
from abm.simulation.sims import Simulation


def test_init_simulation():
    """
    Testing simulation
    """
    test_env_file = Path(__file__).parent / 'data' / 'test.env'

    playground_params = PlaygroundParameters(_env_file=test_env_file)

    # merge all parameters into one dict
    kwargs = {
        **playground_params.environment.dict(),
        **playground_params.ui.dict(),
        **playground_params.agent.dict(),
        **playground_params.resource.dict()
    }

    n = kwargs.pop('n')
    t = kwargs.pop('t')

    sim = Simulation(n, t, **kwargs)
