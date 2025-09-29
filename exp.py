import numpy as np
import matplotlib.pyplot as plt
import lpr 


def generate_random_exp(mfp, n_points=int(1e6)):
    '''Generates 'n_points' with a period ~(modulo) on a pseudo-random dist'''
    points = lpr.generate_random_uniform(n_points) 

    return -mfp * np.log(1. - points)


if __name__ == "__main__":
    npts  = int(1e6)
    nbins = 150 

    mean_free_path = 10. 

    points = generate_random_exp(mean_free_path, npts)

    print("generating", npts, "pts...", end='')
    counts_x, all_bins = np.histogram(points, 150, [0., 50.])
    print("done.")

    bins_x = all_bins[:len(all_bins)-1]

    #plt.ylim([0., (npts/nbins)*1.30])

    plt.plot(bins_x, counts_x)

    plt.show()
