from abm.replay.replay import ExperimentReplay
import numpy as np
import matplotlib.pyplot as plt

loaded_experiment = ExperimentReplay("/home/david/Desktop/database/figExp0NoColl/figExp0N25NoColl")
loaded_experiment.start()
# loaded_experiment.experiment.calculate_search_efficiency()
# m_eff = np.mean(loaded_experiment.experiment.efficiency, axis=0)
# print(m_eff.shape)
# for ai in range(5):
#     plt.figure()
#     plt.imshow(m_eff[:, :, 0, ai])
#     plt.show()