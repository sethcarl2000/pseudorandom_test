import numpy as np
import exp
import lpr

def compute_normal_xy(_npts, _mean=0., _stddev=1.):
    '''Compute 2 normally-distributed random numbers sttdev''' 

    _u_list = lpr.generate_random_uniform(_npts)
    _v_list = lpr.generate_random_uniform(_npts)

    _x_list = np.sqrt( -2.*np.log(_u_list) ) * np.cos( 2.*np.pi*_v_list )
    _y_list = np.sqrt( -2.*np.log(_v_list) ) * np.sin( 2.*np.pi*_u_list )

    _x_list = (_x_list * _stddev) + _mean
    _y_list = (_y_list * _stddev) + _mean
    
    return _x_list, _y_list


if __name__ == "__main__":
    