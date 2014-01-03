#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def run(distances_file, npz_file, npz_out_file, fig_out, n_bins, threshold, latex):

    if distances_file is not None:
        distances = np.load(distances_file, mmap_mode='r')

        # distances are saved in float32, force float64
        mean = np.mean(distances, dtype=np.float64)
        std = np.std(distances, dtype=np.float64)

        n, bins, patches = plt.hist(distances, n_bins, normed=False, facecolor='green', alpha=0.5)

        if npz_out_file is not None:
            np.savez(npz_out_file, mean=mean, std=std, n=n, bins=bins, patches=patches)

    elif npz_file is not None:
        archive = np.load(npz_file)

        mean = archive['mean']
        std = archive['std']

        n, bins, patches = archive['n'], archive['bins'], archive['patches']
        n_bins = n.shape[0]
    else:
        print('Please provide distances or archive')
        exit()

    length = sum(n)

    normal_distribution = mlab.normpdf(bins, mean, std ** 2)
    normal_distribution /= normal_distribution.sum()

    # compute theoretical frequencies
    frequencies = normal_distribution * length

    # use only bins with theoretical frequency > threshold
    valid_bins = np.where(frequencies > threshold)

    # compute chi-square contributions
    T = [(n_i - frequencies[valid_bins][i]) ** 2 / frequencies[valid_bins][i] for i, n_i in enumerate(n[valid_bins])]

    print('Mean: {}\nStandard deviation: {}\nChi^2: {}\n\n'.format(mean, std, sum(T)))

    if latex:
        sys.stdout.write('\\begin{table}[htp]\n')
        sys.stdout.write('\t\\footnotesize\n')
        sys.stdout.write('\t\caption*{$\chi^2$ test}\n')
        sys.stdout.write('\t\centering\n')

        sys.stdout.write('\t\\begin{tabular}{|l|')
        for i in valid_bins[0]:
            sys.stdout.write('l|')
        sys.stdout.write('|l|}\n')
        print('\t\hline')

        sys.stdout.write('\t ')
        for i in valid_bins[0]:
            sys.stdout.write('& {} '.format(i))
        sys.stdout.write('& \\\\ \hline\n')

        for _title, _list, _sum, _isfloat in [
                ('N', normal_distribution[valid_bins], 1, True),
                ('X', n[valid_bins], length, False),
                ('teoretická četnost', frequencies[valid_bins], length, False),
                ('příspěvek k $\chi^2$', T, sum(T), False)
                ]:
            sys.stdout.write('\t{}'.format(_title))
            for x in _list:
                if _isfloat:
                    sys.stdout.write('& %.3f ' % x)
                else:
                    sys.stdout.write('& %d ' % x)
            sys.stdout.write('& {} \\\\ \hline\n'.format(_sum))

        sys.stdout.write('\t\end{tabular}\n')
        sys.stdout.write('\end{table}\n')

    normal_distribution_plot = mlab.normpdf(bins, mean, std ** 2)
    normal_distribution_plot /= normal_distribution_plot.sum()
    frequencies_plot = normal_distribution_plot * sum(n)
    plt.plot(bins, frequencies_plot, 'b')

    bin_width = (max(bins) - min(bins)) / n_bins
    for b in valid_bins[0]:
        plt.axvspan(bins[0] + b * bin_width, bins[0] + (b + 1) * bin_width, facecolor='y', linewidth=0, alpha=0.3)

    plt.xlabel('Distance')
    plt.ylabel('Frequency')

    if fig_out is not None:
        plt.savefig(fig_out)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Pairwise distances analysis")

    parser.add_argument('-d', '--distances', help='Numpy binary file containing pairwise distances')
    parser.add_argument('-z', '--npz', help='Numpy archive containing mean, standard deviation and bin counts')
    parser.add_argument('-o', '--npz-out', help='Numpy archive output filename')
    parser.add_argument('-f', '--fig-out', help='Figure filename. Figure will be displayed if not set.')
    parser.add_argument('-b', '--bins', help='Number of bins used for discretization', type=int, default=100)
    parser.add_argument('-t', '--threshold', help='Minimum bin size threshold', type=int, default=100)
    parser.add_argument('-l', '--latex', help='Print info in latex format', action='store_true', default=False)

    args = parser.parse_args()
    run(args.distances, args.npz, args.npz_out, args.fig_out, args.bins, args.threshold, args.latex)
