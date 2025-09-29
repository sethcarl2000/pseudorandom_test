import numpy as np 
import matplotlib.pyplot as plt


def compute_moment(_points, _m):
    '''Compute the m-th moment of the dist '''
    _ret=0 
    for _x in _points: 
        _ret += _x ** _m

    return _ret / len(_points)


def compute_mean_stddev(_points): 
    '''return the mean and stddev of (points)'''
    _avg = np.mean(_points) 
    _stddev = np.sqrt( np.sum(np.pow(_points - _avg, 2)) / len(_points) )

    return _avg, _stddev


def generate_random_uniform(_n_points):
    '''Generates 'n_points' with a period ~(modulo) on a pseudo-random dist'''
    _modulo = 8388593
    _mult   = 653276

    _points = np.empty(_n_points) 

    _val = int(_modulo/2)

    for i in range(0, _n_points): 
        _points[i] = _val/_modulo 
        _val = (_mult * _val) % _modulo

    return _points


# if this function is executed directly
if __name__ == "__main__":

    npts  = int(1.e6)
    nbins = int(150)

    points_uniform = np.random.uniform(0., 1., npts)
    points = generate_random_uniform(npts)

    pts_avg, pts_stddev = compute_mean_stddev(points)
    print("average:", pts_avg, "- stddev:", pts_stddev)


    print("generating", npts, "pts...", end='')
    counts_x, all_bins = np.histogram(points, 150, [0., 1.])
    print("done.")

    bins_x = all_bins[:len(all_bins)-1]

    plt.ylim([0., (npts/nbins)*1.30])

    plt.plot(bins_x, counts_x)

    plt.show()

    for m in range(1, 4+1):
        print("moment",m,":", compute_moment(points,m))


