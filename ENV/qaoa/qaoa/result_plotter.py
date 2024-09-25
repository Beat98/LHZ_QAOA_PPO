import matplotlib.pyplot as plt
import numpy as np

from lhz.spinglass_utility import create_qutip_wf, all_confs


def plot_result(result, title='qaoa result', size=None, ylabels=None):
    # just for compatibility, use plot results
    plot_results([result], title=title, size=size, ylabels=ylabels)


def plot_results(results, title='qaoa result',
                 ylabels=None, legend=None, size=None,
                 skip_measurements=None, filename=None, show=True):
    from qaoa.QaoaBase import Result
    if type(results) is not list and type(results) is Result:
        results = [results]

    if skip_measurements is None:
        skip_measurements = []

    if ylabels is None and results[0].measurement_labels is not None:
        ylabels = ['objective val.'] + results[0].measurement_labels

    plt.figure()
    if results[0].measurements is None:
        num = 1
    else:
        num = 1 + len(results[0].measurements) - len(skip_measurements)
    if size is None:
        size = (9, 1 + 1.2 * num)
    plt.subplots(num, 1, sharex=True, figsize=size)
    plt.subplot(num, 1, 1)
    plt.title(title)

    if legend is None:
        if len(results) <= 5:
            legend = ['result %d' % d for d in range(len(results))]

    for result in results:
        plt.plot(result.objective_values)
    if ylabels is not None:
        plt.ylabel(ylabels[0])
    else:
        plt.ylabel('objective value')

    if legend is not None:
        plt.legend(legend)

    # num = 1 + len(results[0].measurements)
    plotcount = 0
    for i in range(num - 1 + len(skip_measurements)):
        if i in skip_measurements:
            continue
        plt.subplot(num, 1, 2 + plotcount)
        for result in results:
            plt.plot(result.measurements[i])
        if ylabels is not None:
            plt.ylabel(ylabels[i + 1])
        else:
            plt.ylabel('series ' + str(plotcount))
        plotcount += 1
    plt.xlabel('MC step')

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()


def plot_parameters(results, title='qaoa parameters', size=None):
    from qaoa.QaoaBase import Result
    if type(results) is not list and type(results) is Result:
        results = [results]

    plt.figure()
    if size is None:
        size = (9, 2.2)
    plt.subplots(1, 1, figsize=size)
    plt.subplot(1, 1, 1)
    plt.title(title)

    for result in results:
        plt.plot(result.parameters)
    plt.ylabel('parameters')

    legend = ['result %d' % d for d in range(len(results))]
    plt.legend(legend)

    plt.xlabel('i')
    plt.show()


def print_comp_basis(wf, mode='overlap', thresh=0.01, n_digits=3):
    retstrs = []
    for st in all_confs(len(wf.dims[0])):
        qwf = create_qutip_wf(st)
        olap = qwf.overlap(wf)
        prob = abs(qwf.overlap(wf)) ** 2
        if prob >= thresh:
            if mode == 'overlap':
                retstrs += [str(np.round(olap, n_digits)) + '|' + str(st)[1:-1] + '>']
            else:
                retstrs += [str(np.round(prob, n_digits)) + '|' + str(st)[1:-1] + '>']

    retstr = ' + '.join(retstrs)
    return retstr
