import numpy as np
import exp
import lpr
import ROOT as r

def compute_normal_xy(npts, mean=0., sigma=1.):
    '''Compute 2 normally-distributed random numbers sttdev
        (Using box-mueller alg.)
    ''' 

    _u_list = np.random.uniform(0., 1., npts) #lpr.generate_random_uniform(npts)
    _v_list = np.random.uniform(0., 1., npts) #lpr.generate_random_uniform(npts)

    _x_list = np.empty(npts)
    _y_list = np.empty(npts)

    for i in range(0, npts):
        _x_list[i] = np.sqrt( -2.*np.log(_u_list[i]) ) * np.cos( 2.*np.pi*_v_list[i] )
        _y_list[i] = np.sqrt( -2.*np.log(_u_list[i]) ) * np.sin( 2.*np.pi*_v_list[i] )
    
    _x_list = (_x_list * sigma) + mean
    _y_list = (_y_list * sigma) + mean
    
    return _x_list, _y_list


def compute_normal_correlated(npts, sigma_x=1., sigma_y=1., correlation=0.):
    '''
    Return a list of number pairs [x,y], which have given correlation coeff.
    
    This is done by considering that y_i = alpha * x_i + delta_i 
    Where: 
        x_i     = is normally distributed
        alpha   = some const. 
        delta_i = is also independently, normally distributed.

        if we choose our values of [correlation], [<x^2>] and [<y^2>], you can show: 
        alpha       = corrrelation * sqrt(<y^2>/<x^2>) 
        <delta^2>   = <y^2> - alpha^2 * <x^2> 
                    = sigma_y * sqrt(1 - correlation^2)

    '''
    _alpha = correlation * sigma_y / sigma_x 
    _sigma_d = sigma_y * np.sqrt(1 - correlation**2)

    _x,_delta = compute_normal_xy(npts, mean=0., sigma=1.)

    _y = (_alpha*_x) + (_delta*_sigma_d)

    return np.stack([_x,_y], axis=-1)



def make_h2d(data, str_name="h", str_title="", nbinsx=200, xrange=[-10.,10.], nbinsy=200, yrange=[-10.,10.]):
    '''Make a 2d ROOT histogram'''
    _hist = r.TH2D(str_name, str_title, nbinsx, xrange[0], xrange[1], nbinsy, yrange[0], yrange[1])
    
    for _xy in data: 
        _hist.Fill(_xy[0], _xy[1])
    
    return _hist

def print_covariance(data):
    #assumes mean is zero
    for _xy in data:
        


if __name__ == "__main__":
    
    npts = int(1e6)

    pts_05      = compute_normal_correlated(npts, sigma_x=1., sigma_y=2., correlation=+0.5)
    hist_05     = make_h2d(pts_05, "h_05", "#rho = +0.5;x;y")

    pts_pos1    = compute_normal_correlated(npts, sigma_x=1., sigma_y=2., correlation=+1.0)
    hist_pos1   = make_h2d(pts_pos1, "h_p1", "#rho = +1.0;x;y")

    pts_neg1    = compute_normal_correlated(npts, sigma_x=1., sigma_y=2., correlation=-1.0)
    hist_neg1   = make_h2d(pts_neg1, "h_n1", "#rho = -1.0;x;y")

    pts_zero    = compute_normal_correlated(npts, sigma_x=1., sigma_y=2., correlation=0.)
    hist_zero   = make_h2d(pts_zero, "h_zero", "#rho = 0;x;y")

    canv = r.TCanvas("canv", "test of gaussian generator")

    canv.Divide(2,2)
    
    r.gStyle.SetOptStat(0)

    canv.cd(1); hist_05.Draw("col")
    canv.cd(2); hist_pos1.Draw("col")
    canv.cd(3); hist_neg1.Draw("col")
    canv.cd(4); hist_zero.Draw("col")

    canv.SaveAs("2d_gauss.png")
