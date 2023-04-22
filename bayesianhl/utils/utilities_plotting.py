import numpy as np
import matplotlib.pyplot as plt


def visualize_query_distrn(A_cr, p_query, cmax=0.15, FLAG_indicate_query_space=True,
                           save_plot=False, title_plot=None):
    """
    :param p_query: query distribution
    :param save_plot:
    :param title_plot:
    FLAG_indicate_query_space: whether to indicate the usual query space used
    :return:
    """
    q_plot = p_query.reshape([A_cr.N_actions_t, A_cr.N_actions_M * A_cr.N_actions_U]).T

    plt.figure(2, figsize=(14, 6))
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    img = plt.imshow(q_plot, interpolation='nearest', aspect='auto')
    img.set_cmap('Blues')
    plt.colorbar()
    plt.clim(0, cmax)

    plt.xlabel(r"t ($\times 10^{-7}$s)")
    plt.ylabel(r"($M,U$)")

    if FLAG_indicate_query_space:
        ylabels_plot = [r'', r'($M_{\langle X \rangle}, U_0$)', r'($M_{\langle X \rangle}, U_1$)',
                        r'($M_{\langle Y \rangle}, U_0$)', r'($M_{\langle Y \rangle}, U_1$)',
                        r'($M_{\langle Z \rangle}, U_0$)', r'($M_{\langle Z \rangle}, U_1$)']

        fig_axis = plt.gca()
        fig_axis.set_yticklabels(ylabels_plot)

        xlocs, xlabels = plt.xticks()  # Get locations and labels

        # xlabels has empty strings in the very beginning and end
        # xlocs includes the leftmost and rightmost

        n_xlabels = 10
        xlocs = np.linspace(0, len(A_cr.tset) - 1, n_xlabels, dtype=int)
        time_info = A_cr.tset[xlocs]
        # xlabels_plot = [r'']
        xlabels_plot = []
        for ind in range(len(time_info)):
            xlabels_plot.append(str(np.round(time_info[ind], 1)))

        # xlabels_plot.append(r'')
        plt.xticks(xlocs, xlabels_plot)  # Set locations and labels

    if save_plot:
        if title_plot is None:
            print('Title not provided, saving as viz1.eps')
            title_plot = 'viz1.eps'

        plt.savefig(title_plot, bbox_inches='tight')

    plt.show()