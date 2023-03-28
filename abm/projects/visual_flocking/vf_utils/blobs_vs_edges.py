import matplotlib.pyplot as plt
import numpy as np

# creating figure and axes
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

# creating x axis
axis_len = 2000
phi = np.arange(-np.pi, np.pi, (2*np.pi)/2000)

# creating visual blobs
blob_widths = [600, 150]
offsets = [0, int(axis_len/4)]
centers = [int(axis_len/2) + offset for offset in offsets]

# define trigonometric functions
trigs = [np.cos, np.sin]

# define colors
attr_colors = ["green", "blue"]
repul_colors = ["red", "orange"]

for casei, case in enumerate(["front-back", "left-right"]):
    for disti, dis in enumerate(["near", "far"]):
        # choosing axis
        showax = ax[casei, disti]
        plt.axes(showax)

        # creating blobs
        VPF = np.zeros(axis_len)
        blob_width = blob_widths[disti]
        center = centers[casei]
        bstart = center - int(blob_width/2)
        bend = center + int(blob_width/2)
        VPF[bstart:bend] = 1

        # creating edges
        VPF_edge = np.zeros(axis_len)
        VPF_edge[bstart] = 1
        VPF_edge[bend] = 1

        # modulating with trig function
        VPF_mod = VPF * trigs[casei](phi)
        VPF_edge_mod = VPF_edge * trigs[casei](phi)

        # plotting
        plt.fill_between(phi, np.zeros(axis_len), VPF_mod, alpha=0.5, color=repul_colors[casei])
        plt.plot(phi, VPF, c="gray", ls="-")
        plt.plot(phi, trigs[casei](phi), c="gray", ls="--")

        # plt.plot(phi, VPF_edge_mod, c="green")
        plt.arrow(phi[bstart], 0, 0, trigs[casei](phi[bstart]), width=0.05, length_includes_head=True, edgecolor=attr_colors[casei], facecolor=attr_colors[casei])
        plt.arrow(phi[bend], 0, 0, trigs[casei](phi[bend]), width=0.05, length_includes_head=True, edgecolor=attr_colors[casei], facecolor=attr_colors[casei])
        # plt.plot(phi, VPF_mod)


plt.subplots_adjust(hspace=0.2, wspace=0)
plt.show()