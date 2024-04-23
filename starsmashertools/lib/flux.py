# I hate how this is written, but at least it works specifically for us...
import starsmashertools.preferences
import math
import numpy as np
import starsmashertools.lib.output
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer
import copy
import warnings
import typing

try:
    import matplotlib.axes
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

class Null(object): pass

def process_inputs(**kwargs):
    prefs = get.preferences.get('options')
    for key, val in kwargs.items():
        if isinstance(val, Null):
            if key in prefs.keys(): kwargs[key] = prefs[key]
            elif key not in ['dust_opacity']:
                raise ValueError("Keyword argument '%s' must be given if it is not featured in the preferences files" % key)
    return kwargs



@starsmashertools.preferences.use
def get(
        output : starsmashertools.lib.output.Output,
        weighted_averages : list | tuple = [],

        # Image
        resolution : list | tuple | type(Null) = Null(),
        extent : list | tuple | type(None) = None,
        extent_limits : list | tuple | type(None) = None,
        theta : int | float | type(Null) = Null(),
        phi : int | float | type(Null) = Null(),
        fluffy : bool | type(Null) = Null(),
        rays : bool | type(Null) = Null(),
        tau_s : float | int | type(Null) = Null(),
        tau_skip : float | int | type(Null) = Null(),
        teff_cut : float | int | type(Null) = Null(),
        dust_opacity : float | int | type(None) | type(Null) = Null(),
        dust_Trange : list | tuple | type(Null) = Null(),
        flux_limit_min : int | float | type(None) | type(Null) = Null(),

        # Spectrum
        spectrum_size : int | float | type(Null) = Null(),
        lmax : int | float | type(Null) = Null(),
        dl_il : int | float | type(Null) = Null(),
        dl : int | float | type(Null) = Null(),
        nbox : int | float | type(Null) = Null(),
        nboxh : int | float | type(Null) = Null(),
        l_range : int | float | type(Null) = Null(),
        factor1 : int | float | type(Null) = Null(),
        factor2 : int | float | type(Null) = Null(),
        filters : dict | type(Null) = Null(),

        # Debugging
        verbose : bool = False,
        test_ray : bool = False,
        itr : int = -1,
        jtr : int = -1,
        test_particle : bool = False,
        ic : int = -1,
):
    
    # Obtain all of the keywords (only) above
    kwargs = locals()
    kwargs.pop('output')

    import starsmashertools

    kwargs = process_inputs(**kwargs)
    
    resolution = kwargs['resolution']
    extent = kwargs['extent']
    extent_limits = kwargs['extent_limits']
    theta = kwargs['theta']
    phi = kwargs['phi']
    
    do_fluffy = kwargs['fluffy']
    do_dust = kwargs['dust_opacity'] is not None
    do_rays = kwargs['rays']

    tau_s = kwargs['tau_s']
    tau_skip = kwargs['tau_skip']
    teff_cut = kwargs['teff_cut']
    kappa_dust = kwargs['dust_opacity']
    T_dust_min, T_dust = kwargs['dust_Trange']
    flux_limit_min = kwargs['flux_limit_min']

    spectrum_size = kwargs['spectrum_size']
    lmax = kwargs['lmax']
    l_min, l_max = kwargs['l_range']
    dl = kwargs['dl']
    il_max = math.floor(lmax / dl)
    n_l = math.floor((l_max - l_min) / kwargs['dl_il']) + 1
    nbox = kwargs['nbox']
    nboxh = kwargs['nboxh']
    factor1 = kwargs['factor1']
    factor2 = kwargs['factor2']
    filters = kwargs['filters']
    
    # Avoid headaches with the weighted averages
    weighted_averages = [copy.deepcopy(w) for w in weighted_averages]
    
    current = copy.deepcopy(output)
    
    # Set viewing angle
    current.rotate(
        xangle = theta,
        zangle = phi,
    )
    
    simulation = current.simulation

    # All in cgs
    c_speed = float(simulation.units.constants['c'])
    sigma_const = float(simulation.units.constants['sigmaSB'])
    a_const = float(simulation.units.constants['a'])
    
    
    # in case, to make sure we use the same units
    msun = float(simulation.units.mass)
    rsun = float(simulation.units.length)
    
    ntot = current['ntot']
    time = current['t'] * simulation.units['t']
    rho = current['rho'] * float(simulation.units['rho'])
    m = current['am']
    h = current['hp']
    u = current['u'] * float(simulation.units['u'])

    uraddot = current['uraddot'] * float(simulation.units['uraddot'])
    if simulation['ncooling'] == 2: # No heating
        uraddotcool = copy.deepcopy(uraddot)
        uraddotheat = np.zeros(uraddot.shape)
    elif simulation['ncooling'] == 3: # Heating
        uraddotcool = current['uraddotcool'] * float(simulation.units['uraddotcool'])
        uraddotheat = current['uraddotheat'] * float(simulation.units['uraddotheat'])
    else:
        raise NotImplementedError("ncooling = %d is not supported" % simulation['ncooling'])
    
    uraddot_emerg = current['dEemergdt'] * float(simulation.units['dEemergdt'])
    uraddot_diff = current['dEdiffdt'] * float(simulation.units['dEdiffdt'])
    kappa = current['popacity'] * float(simulation.units['popacity'])
    tau = current['tau']
    temp = current['temperatures']
    rloc=np.full(ntot, np.nan)
    x=current['x']
    y=current['y']
    z=current['z']

    
    cores = u == 0


    
    # Spectrum setup
    spectrum = np.zeros(spectrum_size, dtype=float)
    b_l = np.zeros(spectrum.shape, dtype=float)

    lam = (l_min + dl*np.arange(n_l))*1e-7 # to nanometers
    invlam = 1. / lam
    invlam5 = invlam**5
    
    spect_min = np.zeros(ntot, dtype=int)
    spect_max = np.zeros(spect_min.shape, dtype=int)
    spect_type = np.zeros(spect_min.shape, dtype=int)
    if verbose: print("Spectrum:", n_l, dl)

    #======
    
    
    if extent is None:
        rloc_arr = np.full(ntot, np.nan)
        if cores.any():
            rloc_arr[cores] = 2 * h[cores]
        if (~cores).any():
            rloc_arr[~cores] = (0.75*m[~cores]*msun/(np.pi*rho[~cores]))**0.33333/rsun
        if do_fluffy:
            rloc_arr = 2.*h
        xmin = np.amin(x - rloc_arr)
        xmax = np.amax(x + rloc_arr)
        ymin = np.amin(y - rloc_arr)
        ymax = np.amax(y + rloc_arr)

        max_coord=max(abs(xmin),abs(xmax), abs(ymin), abs(ymax))
        xmax=max_coord
        ymax=max_coord
        ymin=-max_coord
        xmin=-max_coord    
        if verbose: print("domain  max: %12.5f" %  max_coord)
    else:
        xmin, xmax, ymin, ymax = extent
    
    if extent_limits is not None:
        xminlim, xmaxlim, yminlim, ymaxlim = extent_limits
        if xminlim is not None: xmin = max(xmin, xminlim)
        if xmaxlim is not None: xmax = min(xmax, xmaxlim)
        if yminlim is not None: ymin = max(ymin, yminlim)
        if ymaxlim is not None: ymax = min(ymax, ymaxlim)
    
    if verbose:
        print("xmin: %12.5f" % xmin)
        print("xmax: %12.5f" % xmax)
        print("ymin: %12.5f" % ymin)
        print("ymax: %12.5f" % ymax)


    
        
        

    if verbose: print("Time = %s" % time)
    
    rows, cols = np.asarray(resolution) + 1
    shape = (rows, cols)
    surf_d = np.zeros(shape)
    surf_id = np.zeros(shape,dtype=int)
    surf_br = np.zeros(shape)
    surf_br_v = np.zeros(shape)
    
    weighted_average_arrays = []
    for i in range(len(weighted_averages)):
        weighted_average_arrays += [np.zeros(surf_br.shape)]

    kappa_cell = np.zeros(shape, dtype = float)
    rho_cell = np.zeros(shape, dtype = float)
    T_cell = np.zeros(shape, dtype = float)
    surf_t = np.zeros(shape, dtype = float)
    flux = np.zeros(ntot, dtype = float)
    teff = np.zeros(ntot, dtype = float)
    
    # This method is equivalent to creating an empty numpy array of type list
    # with shape (rows, cols) and then filling the array with list() instances,
    # e.g.:
    #    ray_id = np.empty((rows,cols), dtype=list)
    #    for i in range(rows):
    #        for j in range(cols):
    #           ray_id[i][j] = list()
    # However, this method is much, much faster.
    # https://stackoverflow.com/a/33987165
    ray_id = np.frompyfunc(list, 0, 1)(np.empty(shape, dtype = object))

    ray_n = np.zeros(shape, dtype = int)
    
    # Helpful array for storing in the output files for debugging later on
    contributors = np.full(flux.shape, False, dtype = bool)
    
    surf_id[:] = -1
    surf_d[:] = -1.e30

    grid_indices = np.indices(shape).transpose(1,2,0)
    
    dx = (xmax-xmin)/resolution[0]
    dy = (ymax-ymin)/resolution[1]

    if verbose: print("dx=%f dy=%f"%(dx,dy))
    
    if do_dust:
        kappa = set_dust(current, kappa_dust, temp, T_dust_min, T_dust)

    # Below is a much faster alternative to looping through particles to set
    # values.
    flux_rad = np.zeros(flux.shape)
    flux_em = np.zeros(flux.shape)

    idx = np.full(ntot, False)
    if do_fluffy:
        idx = np.logical_and(uraddot_emerg < uraddot_diff)
    
    i = np.logical_and(idx, ~cores)
    if i.any():
        rloc[i] = 2.*h[i]
        rho[i]=m[i]*msun/pow(rloc[i]*rsun,3.)
        tau[i]=rho[i]*kappa[i]*rloc[i]*rsun*4./3.
        flux_em[i] = uraddot_emerg[i]*0.25/(np.pi*(rloc[i]*rsun)**2)
        flux[i]=flux_em[i]
        flux_rad[i] = flux_em[i]

    i = np.logical_and(~idx, ~cores)
    if i.any():
        # this particle is considered in "dense" approximation
        rloc[i]=pow(m[i]*msun/(4/3.*np.pi)/rho[i],0.33333)/rsun
        tau[i]=rho[i]*kappa[i]*rloc[i]*rsun*4./3.
        ad      = 4.*np.pi*(rloc[i]*rsun)**2
        ad_cool = 2.*np.pi*(rloc[i]*rsun)**2
        flux_em[i] = uraddot_emerg[i]/ad
        dd=c_speed/kappa[i]/rho[i]
        tdiffloc=kappa[i]*rho[i]*(rloc[i]*rsun)**2/c_speed
        gradt=a_const*pow(temp[i],4)/(rloc[i]*rsun)
        dedt=ad/3.*dd*gradt/(m[i] * msun)
        flux[i]=dedt*m[i]*msun/ad
        flux_rad[i] = dedt * m[i] * msun / ad

        if do_rays:
            
            ratio = np.full(flux.shape, np.nan)
            ratio[i] = -uraddotcool[i]/dedt*12. # to figure out how many effective cooling rays
            _idx1 = np.logical_and(
                ~cores,
                np.logical_and(i, ratio <= 6),
            )
            if _idx1.any():
                ad_cool = 2.*np.pi*rloc[_idx1]*rloc[_idx1]*rsun*rsun
                flux[_idx1]=-uraddotcool[_idx1]*m[_idx1]*msun/ad_cool

                
            _idx2 = np.logical_and(
                ~cores,
                np.logical_and(i, ratio > 6),
            )
            if _idx2.any():
                ad      = 4.*np.pi*rloc[_idx2]*rloc[_idx2]*rsun*rsun
                gradt = a_const*pow(temp[_idx2],4)/rloc[_idx2]/rsun
                dd = c_speed/kappa[_idx2]/rho[_idx2]
                dedt = ad/3.*dd*gradt/m[_idx2]/msun
                flux[_idx2] = dedt*m[_idx2]*msun/ad

        _idx = np.logical_and(
            ~cores,
            np.logical_and(i, tau < 100),
        )
        if _idx.any():
            flux[_idx] = np.minimum(flux[_idx], flux_em[_idx])

    
    teff[~cores] = (flux_rad[~cores] / sigma_const)**0.25
    
    spect_type[~cores] = 1 # black body
    i = np.logical_and(
        ~cores,
        np.logical_and(flux_rad > flux_em, tau < 100),
    )
    if i.any():
        teff[i] = temp[i]
        spect_type[i] = 2 # emerging

    i = np.logical_and(
        ~cores,
        flux > 0,
    )
    if i.any():
        lam_max = 2.9 * 1000000. / teff[i]
        i_lam = (lam_max / 3.).astype(int) - nboxh
        upper = np.full(i_lam.shape, il_max - 1 - nbox, dtype=int)
        i_lam = np.minimum(np.maximum(i_lam, 0), upper) # clamp
        spect_min[i] = i_lam
        spect_max[i] = i_lam + nbox

    # These are used throughout for much faster processing
    ijloc_arr = np.column_stack((
        ((x - xmin) / dx).astype(int), # iloc
        ((y - ymin) / dy).astype(int), # jloc
    ))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ijminmax_arr = np.column_stack((
            # imin
            np.maximum(
                ((x - rloc - xmin) / dx).astype(int), # same as floor
                np.zeros(ntot, dtype = int),
            ),
            # imax
            np.minimum(
                ((x + rloc - xmin) / dx).astype(int) + 1, # same as ceil
                np.full(ntot, resolution[0], dtype = int),
            ),
            # jmin
            np.maximum(
                ((y - rloc - ymin) / dy).astype(int), # same as floor
                np.zeros(ntot, dtype = int),
            ),
            # jmax
            np.minimum(
                ((y + rloc - ymin) / dy).astype(int) + 1, # same as ceil
                np.full(ntot, resolution[1], dtype = int),
            ),
        ))
    

    # Omit ijloc that are outside the image
    inside_image = np.logical_and(
        np.logical_and(ijloc_arr[:,0] >= 0, ijloc_arr[:,0] <= resolution[0]),
        np.logical_and(ijloc_arr[:,1] >= 0, ijloc_arr[:,1] <= resolution[1]),
    )
    if not inside_image.any():
        raise Exception("There are no particles located within the given extents")
    
    
    #for i in np.arange(ntot)[~cores]:
    for i, (iloc, jloc) in enumerate(ijloc_arr):
        if inside_image[i]:
            if tau[i] >= tau_s and z[i] > surf_d[iloc][jloc] :
                imin, imax, jmin, jmax = ijminmax_arr[i]
                
                idx = z[i] > surf_d[imin:imax, jmin:jmax]
                if not idx.any(): continue

                xloc_arr = xmin + dx * np.arange(imin, imax)
                yloc_arr = ymin + dy * np.arange(jmin, jmax)
                deltax2_arr = (xloc_arr - x[i])**2
                dist2_arr = deltax2_arr[:, None] + (yloc_arr - y[i])**2

                idx = np.logical_and(idx, dist2_arr <= rloc[i]**2)
                if not idx.any(): continue
                
                surf_d[imin:imax, jmin:jmax][idx] = z[i]
                surf_id[imin:imax, jmin:jmax][idx] = i


    if verbose and test_ray:
        print("created a ray to %d %d  id=%d z=%8.4e tau=%8.4e    %e %e"%(itr,jtr,surf_id[itr][jtr],surf_d[itr][jtr],tau[surf_id[itr][jtr]], xmin+dx*itr,ymin+dy*jtr))
        #exit()

    # Moved verbose printing outside the calculations to speed things up
    if verbose and test_particle and ic in np.arange(ntot):
        iloc, jloc = ijloc_arr[ic]
        if inside_image[ic]:
            print("CHECK 0 %d %d %d z=%8.4e in [%8.4e; %8.4e] x=%8.4e y=%8.4e tau=%8.4e surface=%8.4e %8.4e %d"% (ic, iloc, jloc, z[ic], z[ic]-rloc[ic], z[ic]+rloc[ic], x[ic], y[ic], tau[ic],  surf_d[iloc][jloc], z[surf_id[iloc][jloc]], surf_id[iloc][jloc]))
            imin,imax,jmin,jmax = ijminmax_arr[ic]

            print("CHECK 1 %d %d %d %d %d z=%8.4e in [%8.4e; %8.4e] x==%8.4e y=%8.4e tau=%8.4e surface=%8.4e %8.4e"% (ic, imin, imax, jmin, jmax, z[ic], z[ic]-rloc[ic], z[ic]+rloc[ic], x[ic], y[ic], tau[ic],  surf_d[ii][jj], z[surf_id[ii][jj]]))

            for ii in range(imin, imax):
                for jj in range(jmin, jmax):
                    print("CHECK 2 %d %d %d z=%8.4e in [%8.4e; %8.4e] x==%8.4e y=%8.4e tau=%8.4e surface=%8.4e %8.4e"% (ic, ii, jj, z[ic], z[ic]-rloc[ic], z[ic]+rloc[ic], x[ic], y[ic], tau[ic],  surf_d[ii][jj], z[surf_id[ii][jj]]))

    tau_min=1

    skip = tau < tau_skip
    if flux_limit_min is not None:
        skip = np.logical_or(skip, flux[i] <= flux_limit_min)
    
    for i, ((iloc, jloc), (imin,imax,jmin,jmax)) in enumerate(zip(ijloc_arr, ijminmax_arr)):
        if inside_image[i]:
            if skip[i]: continue
            
            idx = z[i] > surf_d[imin:imax, jmin:jmax]
            if not idx.any(): continue

            xloc_arr = xmin + dx * np.arange(imin, imax)
            yloc_arr = ymin + dy * np.arange(jmin, jmax)
            deltax2_arr = (xloc_arr - x[i])**2
            deltay2_arr = (yloc_arr - y[i])**2
            dist2_arr = deltax2_arr[:, None] + deltay2_arr

            idx = np.logical_and(idx, dist2_arr <= rloc[i]**2)
            if not idx.any(): continue
            
            tau_min = min(tau_min, tau[i])

            for ii, jj in grid_indices[imin:imax, jmin:jmax][idx]:
                ray_id[ii, jj].append(i)
                ray_n[ii, jj] += 1

    # Much more efficient calculations below
    if verbose:
        max_ray = np.amax(ray_n)
        idx_maxray = np.argmax(ray_n.flatten())
        i_maxray, j_maxray = np.unravel_index(idx_maxray, ray_n.shape)
    
        print("maximum number of particles above the cut off optically thick surface is %d %d %d"%(max_ray,i_maxray,j_maxray))            
        print("minimum tau account for is  is %f"%(tau_min))            


    # this sorting is as simple as hell 
    # Package array in the loop for quicker iterations (moves the elements onto
    # lower level caches)
    for ii, ray_n_ii in enumerate(ray_n[:resolution[0]]):
        for jj, ray_n_iijj in enumerate(ray_n_ii[:resolution[1]]):
            if ray_n_iijj > 1:
                # Pre-build ranges for faster iterations
                ri_range = range(ray_n_iijj - 1)
                for kk in range(ray_n_iijj):
                    swap=0
                    for ri in ri_range:
                        ir1=ray_id[ii][jj][ri]
                        ir2=ray_id[ii][jj][ri+1]
                        if ir1==0 or ir2 ==0:
                            raise Exception("ray to the core? %d %d", ii, jj)
                        if z[ir1] > z[ir2]:
                            ray_id[ii][jj][ri]=ir2
                            ray_id[ii][jj][ri+1]=ir1
                            swap=swap+1
                    if swap == 0:
                        break
                if swap > 0:
                    raise Exception("Did not complete sorting")


    if verbose and test_ray:
        print("ray outwards")  
        print("surface at %d %d   id=%d  z=%e"%(itr,jtr,surf_id[itr][jtr], z[surf_id[itr][jtr]]))
        if ray_n[itr][jtr] > 0:
            for ri in range(0, ray_n[itr,jtr]):
                print("ray at %d %d  id=%d  number of rays %d"%(itr,jtr,ray_id[itr][jtr][ri],ray_n[itr,jtr]))
                print("ray backwards")  
        if ray_n[itr][jtr] > 0:
            for ri in range(ray_n[itr,jtr]-1, -1, -1):
                print("ray at %d %d  id=%d"%(itr,jtr,ray_id[itr][jtr][ri]))
                print("surface at %d %d   id=%d  number of rays %d"%(itr,jtr,surf_id[itr][jtr],ray_n[itr,jtr]))

    # Vectorized array initialization
    surf_br_v[:resolution[0]] = 0
    surf_br[:resolution[0]] = 0

    # Package the loop for quicker enumeration
    zipped = zip(
        surf_id[:resolution[0],:resolution[1]],
        ray_n[:resolution[0],:resolution[1]],
        ray_id[:resolution[0],:resolution[1]],
    )
    
    b_fulls = sigma_const / np.pi * teff**4
    lam5 = lam**5
        
    for ii, (surf_id_row, ray_n_ii, ray_id_ii) in enumerate(zipped):
        _zipped = zip(
            surf_id_row,
            ray_n_ii,
            ray_id_ii
        )
        for jj, (i, ray_n_iijj, ray_id_iijj) in enumerate(_zipped):
            tau_ray=0.    
            # Reversing an array is easy. This produces an iterator ('view'
            # of the list) which runs faster than ray_id[ii][jj][::-1] and
            # also faster than range(ray_n[ii][jj]-1, -1, -1).
            for ir in reversed(ray_id_iijj):
                f = flux[ir] * math.exp(-tau_ray)
                surf_br[ii,jj] += f #flux[ir] * np.exp(-tau_ray)
                contributors[ir] = True

                for _i, A in enumerate(weighted_averages):
                    weighted_average_arrays[_i][ii, jj] += A[ir] * f

                kappa_cell[ii,jj] += kappa[ir] * f
                rho_cell[ii,jj] += rho[ir] * f
                T_cell[ii,jj] += temp[ir] * f
                    
                # Do spectrum stuff
                expon = factor2 / (teff[ir] * lam)
                idx = expon < 150
                _b_l = factor1 / ((np.exp(expon[idx]) - 1) * lam5[idx])
                spectrum[:n_l][idx] += f * (_b_l / b_fulls[ir])
                
                if verbose and test_ray and ii==itr and jj==jtr:
                    print("brigntess modification backward %d  br=%e flux_loc=%e tau_t=%e tau_l=%e   z=%e in [%e,%e] x=%e y=%e"%(ir, surf_br[ii,jj],flux[ir],tau_ray,tau[ir],z[ir],z[ir]-4./3.*h[ir],z[ir]+4./3.*h[ir], x[ir],y[ir]))
                if teff[ir] > teff_cut:
                    surf_br_v[ii][jj] += f #flux[ir] * np.exp(-tau_ray)

                tau_ray += tau[ir]           

                if surf_br_v[ii][jj] >  surf_br[ii][jj]:
                    raise Exception("part of the flux is larger then whole flux?   %d %d %d %d   tau=%8.4e  %8.4e  %8.4e  %8.4e"%(ii,jj,ri,ir,tau[ir],surf_br[ii][jj],surf_br_v[ii][jj],flux[ir]))
            
    
            if i > 0:
                f = flux[i] * math.exp(-tau_ray)
                surf_br[ii,jj] += f #flux[i] * np.exp(-tau_ray)

                for _i, A in enumerate(weighted_averages):
                    weighted_average_arrays[_i][ii, jj] += A[i] * f

                kappa_cell[ii,jj] += kappa[i] * f
                rho_cell[ii,jj] += rho[i] * f
                T_cell[ii,jj] += temp[i] * f
                    
                contributors[i] = True

                expon = factor2 / (teff[i] * lam)
                idx = expon < 150
                _b_l = factor1 / ((np.exp(expon[idx]) - 1) * lam5[idx])
                spectrum[:n_l][idx] += f * (_b_l / b_fulls[i])
                
                if verbose and test_ray and ii==itr and jj==jtr:
                    print("brigntess modification finishes at %d %e %e %e  tau_tot=%e   z=%e in [%e;%e]  x=%e y=%e"%(i, surf_br[ii,jj],flux[i],tau_ray,tau[i],z[i],z[i]-4./3.*h[i],z[i]+4./3.*h[i],x[i],y[i]))
                if teff[i] > teff_cut:
                    surf_br_v[ii][jj] += f #flux[i]* np.exp(-tau_ray)

    if verbose:
        print("rays sorting is completed")
        print("maximum observed Teff", np.amax(teff[contributors]))

        if test_ray:
            print("final brigntess %e "%(surf_br[itr,jtr]))

    # Finish obtaining the weighted averages
    idx = np.logical_and(surf_br > 0, np.isfinite(surf_br))
    if np.any(idx):
        for _i in range(len(weighted_averages)):
            weighted_average_arrays[_i][idx] /= surf_br[idx]
        kappa_cell[idx] /= surf_br[idx]
        rho_cell[idx] /= surf_br[idx]
        T_cell[idx] /= surf_br[idx]
            
    if np.any(~idx):
        for _i in range(len(weighted_averages)):
            weighted_average_arrays[_i][~idx] = 0
        kappa_cell[~idx] = 0
        rho_cell[~idx] = 0
        T_cell[~idx] = 0

    # Vectorized:
    area_br = 0.25 * (\
        surf_br[ :resolution[0]-1, :resolution[1]-1] + \
        surf_br[ :resolution[0]-1,1:resolution[1]] + \
        surf_br[1:resolution[0]  , :resolution[1]-1] + \
        surf_br[1:resolution[0]  ,1:resolution[1]])

    surf_t = (area_br/sigma_const)**0.25
    flux_tot = np.sum(area_br)
    flux_tot_v = np.sum(surf_br_v[:resolution[0]-1,:resolution[1]-1])

    # Vectorized:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        teff_aver = np.sum(surf_t[:resolution[0]-1, :resolution[1]-1] * area_br / flux_tot)

    # Finish the ltot and l_v calculations
    ltot = 4 * dx * dy * rsun**2 * flux_tot
    l_v  = 4 * dx * dy * rsun**2 * flux_tot_v

    # Finish spectrum calculations
    teff_spectrum = 0
    max_flux = 0
    ltot_spectrum = 0
    l_spectrum = {key:0 for key in filters.keys()}
    for il in range(il_max - 1):
        sp = spectrum[il]
        ltot_spectrum += sp * dl * 1e-7
        if sp <= 1: continue
        
        sp = np.log10(sp)
        lamb = lam[il] / 1e-7
        teff_loc = 2.9*1.e6/lamb
        if sp > max_flux:
            max_flux = sp
            teff_spectrum = teff_loc

        for key, (lamb_low, lamb_hi) in filters.items():
            if lamb_low is not None and lamb < lamb_low: continue
            if lamb_hi is not None and lamb > lamb_hi: continue
            
            l_spectrum[key] += spectrum[il] * dl * 1e-7
            

    b_l[:] = 0
    spectrum_output = []
    b_full = sigma_const / np.pi * teff_spectrum**4
    for il in range(il_max - 1):
        sp = spectrum[il]
        if sp <= 1: continue
        sp = np.log10(sp)
        lamb = lam[il] / 1e-7
        teff_loc = 2.9*1.e6/lamb

        expon = factor2 / (teff_spectrum * lam[il])
        if expon < 150:
            b_l[il] = factor1 / (lam[il]**5 * (np.exp(expon) - 1))
        else:
            b_l[il] = 0
        spectrum_output += [[il, lamb, teff_loc, sp, b_l[il]]]
    
    ltot_spectrum *= 4 * dx * dy * rsun**2
    for key, val in l_spectrum.items():
        l_spectrum[key] = val * 4 * dx * dy * rsun**2
    
    # Vectorized alternative:
    Lcool = np.sum(uraddotcool[~cores] * m[~cores])
    Lheat = np.sum(uraddotheat[~cores] * m[~cores])
    Lurad = np.sum(uraddot[~cores] * m[~cores])
    
    if verbose:
        print("Check luminosities ltot=%le ltot_spectrum=%le " % (ltot, ltot_spectrum))
        print("Total flux = %le" % flux_tot)
        print("Averagef Teff = %le" % teff_aver)
        print("Total L = %12.4E" % ltot)
        for key, val in l_spectrum.items():
            print('%s L = %12.4E' % (key, lv_spectrum))
        print("Visual L = %12.4E"% l_v)
        print("Cooling L = %12.4E"% (Lcool*msun))
        print("Heating L = %12.4E"% (Lheat*msun))
        print("udot L = %12.4E"% (Lurad*msun))
    
    return FluxResult({
        'version' : starsmashertools.__version__,
        
        'output' : current.path,
        'kwargs' : kwargs,

        'simulation' : simulation.directory,
        'units' : simulation.units,
        
        'time' : time,
        
        'particles' : {
            'contributing_IDs' : np.arange(len(contributors))[contributors],
            'rloc' : rloc[contributors],
            'tau' : tau[contributors],
            'kappa' : kappa[contributors],
            'flux' : flux[contributors],
            'Lcool' : Lcool * msun,
            'Lheat' : Lheat * msun,
        },
        
        'image' : {
            'flux' : surf_br,
            'flux_v' : surf_br_v,
            'surf_d' : surf_d,
            'surf_id' : surf_id,
            'Teff_cell' : surf_t,
            'kappa_cell' : kappa_cell,
            'rho_cell' : rho_cell,
            'T_cell' : T_cell,
            'extent' : [xmin, xmax, ymin, ymax],
            'dx' : dx,
            'dy' : dy,
            'weighted_averages' : weighted_average_arrays,
            'teff_aver' : teff_aver,
            'ltot' : ltot,
            'flux_tot' : flux_tot,
            'l_v' : l_v,
        },
        
        'spectrum' : {
            'ltot_spectrum' : ltot_spectrum,
            'l_spectrum' : l_spectrum,
            'output' : spectrum_output,
            'teff' : teff_spectrum,
        },
    })


# Vectorized dust handling
def set_dust(output, kappa_dust, temp, T_dust_min, T_dust):
    units = output.simulation.units
    kappa = copy.deepcopy(output['popacity']) * float(units['popacity'])
    idx = find_dusty_particles(
        output,
        T_dust_min = T_dust_min,
        T_dust_max = T_dust,
        kappa_dust = kappa_dust,
    )
    if idx.any():
        kappa[idx] = kappa_dust
    return kappa


def find_dusty_particles(
        output : starsmashertools.lib.output.Output,
        T_dust_min : float | int | type(None) = None,
        T_dust_max : float | int | type(None) = None,
        kappa_dust : float | int | type(None) = None,
):
    units = output.simulation.units
    kappa = output['popacity'] * float(units['popacity'])
    T = output['temperatures'] * float(units['temperatures'])
    
    if kappa_dust is not None:
        if T_dust_max is None and T_dust_min is not None:
            return np.logical_and(T > T_dust_min, kappa < kappa_dust)
        elif T_dust_max is not None and T_dust_min is None:
            return np.logical_and(T < T_dust_max, kappa < kappa_dust)

        if T_dust_max is None and T_dust_min is None: return kappa < kappa_dust
        
        return np.logical_and(
            np.logical_and(T < T_dust_max, T > T_dust_min),
            kappa < kappa_dust,
        )

    if T_dust_max is None and T_dust_min is not None: return T > T_dust_min
    elif T_dust_max is not None and T_dust_min is None: return T < T_dust_max
    
    return np.full(False, T.shape, dtype = bool)
    

















class FluxResult(dict, object):
    def __init__(self, *args, **kwargs):
        self._simulation = None
        self._output = None
        super(FluxResult, self).__init__(*args, **kwargs)
    
    @property
    def simulation(self):
        if self._simulation is None:
            import starsmashertools
            self._simulation = starsmashertools.get_simulation(self['simulation'])
        return self._simulation

    @property
    def output(self):
        if self._output is None:
            self._output = starsmashertools.lib.output.Output(
                self['output'], self.simulation,
            )
        return self._output
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def save(
            self,
            filename : str = 'flux.zip',
            **kwargs
    ):
        """
        Save the results to disk as an :class:`~.lib.archive.Archive`.
        
        Other Parameters
        ----------
        filename : str, default = 'flux.zip'
            The name of the Archive.

        **kwargs
            Keyword arguments are passed directly to 
            :meth:`~.lib.archive.Archive.__init__`.
        
        Returns
        -------
        archive : :class:`~.lib.archive.Archive`
            The newly created Archive.

        See Also
        --------
        :meth:`~.load`
        """
        import starsmashertools.lib.archive
        if filename is None: filename = 'flux.zip'
        kwargs['auto_save'] = False
        archive = starsmashertools.lib.archive.Archive(filename, **kwargs)
        for key, val in self.items():
            archive.add(
                key,
                val,
                origin = self['output'],
            )
            print("Added",key)
        archive.save(verbose = True)
        return archive
    
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def load(filename : str):
        """
        Load results from disk which were saved by :meth:`~.save`.

        Parameters
        ----------
        filename : str
            The location of the file on the disk to load.

        Returns
        -------
        :class:`~.FluxResult`

        See Also
        --------
        :meth:`~.save`
        """
        import starsmashertools.lib.archive
        archive = starsmashertools.lib.archive.Archive(filename)
        return FluxResult(archive.get(archive.keys()))
    
    if has_matplotlib:
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        @api
        def plot(
                self,
                ax : matplotlib.axes.Axes | type(None) = None,
                key : str | typing.Callable = 'flux',
                weighted_average : int | type(None) = None,
                log10 : bool = False,
                **kwargs
        ):
            """
            Creates a new :class:`~.mpl.artists.FluxPlot` and calls its 
            :meth:`~.mpl.artists.FluxPlot.imshow` method on the given Matplotlib
            :class:`matplotlib.axes.Axes`.
            
            Other Parameters
            ----------------
            ax : :class:`matplotlib.axes.Axes`, None, default = None
                The Matplotlib :class:`matplotlib.axes.Axes` to plot on. If 
                `None`, the plot is created on the axes returned by
                ``plt.gca()``.

            key : str, default = 'flux'
                The dictionary key in the ``['image']`` :py:class:`dict` to 
                obtain the data for plotting, or a function which accepts a 
                FluxResult as input and returns a 2D NumPy array and a dict, 
                where the 2D array is the image contents and the dict is the 
                keywords to pass to :func:`matplotlib.axes.Axes.imshow`. If 
                ``weighted_average`` is given, this argument is ignored.

            weighted_average : int, None, default = None
                The integer index of the array to plot from the 
                ``weighted_averages`` key in the results. If `None`, keyword 
                argument ``key`` is ignored.

            log10 : bool, default = False
                If `True`, the log10 operation will be done on the data.
            
            kwargs
                Other keyword arguments are passed directly to 
                :meth:`~.mpl.artists.FluxPlot.imshow`. Note, keyword ``origin`` 
                is always set to ``'lower'`` regardless of any value it has in
                ``kwargs``.

            Returns
            -------
            :class:`matplotlib.axes.AxesImage`, :class:`~.mpl.artists.FluxPlot`
                The :class:`matplotlib.axes.AxesImage` object created by 
                :meth:`~.mpl.artists.FluxPlot.imshow` and the 
                :class:`~.mpl.artists.FluxPlot` object which created it.
            
            See Also
            --------
            :class:`~.mpl.artists.FluxPlot`
            """
            import starsmashertools.mpl.artists
            import matplotlib.pyplot as plt
            
            if ax is None: ax = plt.gca()

            if weighted_average is not None:
                data = self['image']['weighted_averages'][weighted_average]
            else:
                if isinstance(key, str):
                    data = self['image'][key]
                elif callable(key):
                    data, kwargs = key(self)
                else:
                    raise Exception("This should never be possible")

            if log10:
                idx = data < 0
                if idx.any():
                    starsmashertools.helpers.warnings.warn(
                        'Some data is <= 0 and log10 = True. The data which is <= 0 will be set to NaN.',
                    )
                # It's common for the empty cells to have flux = 0, so we step
                # around the warnings by checking for <= 0 here.
                idx = data <= 0
                data[idx] = np.nan
                idx = np.isfinite(data)
                data[idx] = np.log10(data[idx])
                data[~idx] = np.nan
            
            ret = starsmashertools.mpl.artists.FluxPlot(ax, self)
            im = ret.imshow(data, **kwargs)
            return ret, im
