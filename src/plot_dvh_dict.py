import matplotlib.pyplot as plt

def plot_dvhs(ax_to_plot, dvhs):
    for i, (organ, dvh) in enumerate(dvhs.items()):
        ax_to_plot.plot(np.arange(0, int(len(dvh)/10) + 0.1, 0.1), dvh)
        ax_to_plot.set_xlabel('Dose [Gy]')
        ax_to_plot.set_ylabel('Volume [%]')
        ax_to_plot.set_xlim(0, 0.1*len(dvh))
        ax_to_plot.set_ylim(0, 100)