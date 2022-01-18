from abm.metarunner.metarunner import Tunable, Constant, MetaProtocol
from time import sleep

# Defining criterions as parameter ranges (Tunables) and single parameter values (Constants)
criteria = []
criteria.append(Tunable("ENV_WIDTH", 600, 800, 3))
criteria.append(Tunable("ENV_HEIGHT", 700, 800, 2))
criteria.append(Constant("T", 100))
criteria.append(Constant("USE_IFDB_LOGGING", 1))
criteria.append(Constant("SAVE_CSV_FILES", 1))

# Creating metaprotocol and add defined criteria
mp = MetaProtocol()
for crit in criteria:
    mp.add_criterion(crit)

# Generating temporary env files with criterion combinations. Comment this out if you want to contzinue simulating due
# to interruption
mp.generate_temp_env_files()
sleep(2)

# Running the simulations
mp.run_protocols()