from abm.replay.replay import ExperimentReplay

loaded_experiment = ExperimentReplay("/media/david/DMezeySCIoI/ABMData/VisFlock/VFExp4c")
loaded_experiment.start()
# loaded_experiment.experiment.calculate_search_efficiency()
# m_eff = np.mean(loaded_experiment.experiment.efficiency, axis=0)
# print(m_eff.shape)
# for ai in range(5):
#     plt.figure()
#     plt.imshow(m_eff[:, :, 0, ai])
#     plt.show()