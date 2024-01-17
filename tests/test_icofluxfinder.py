import basetest
import starsmashertools
import starsmashertools.flux.icofluxfinder
import os
import numpy as np

# These are the angles directly from the log0.sph file.
angles = np.array([
    [1.57079632679490 ,       2.12437068569194     ],
    [1.57079632679490 ,       1.01722196789785     ],
    [1.57079632679490 ,      -2.12437068569194     ],
    [1.57079632679490 ,      -1.01722196789785     ],
    [0.553574358897045,      -1.57079632679490     ],
    [0.553574358897045,       1.57079632679490     ],
    [2.58801829469275 ,      -1.57079632679490     ],
    [2.58801829469275 ,       1.57079632679490     ],
    [2.12437068569194 ,      0.000000000000000E+000],
    [1.01722196789785 ,      0.000000000000000E+000],
    [2.12437068569194 ,       3.14159265358979     ],
    [1.01722196789785 ,       3.14159265358979     ],
])


def rotate(x, y, z, xangle=0, yangle=0, zangle=0):
    import copy
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    z = copy.deepcopy(z)
    
    # Euler rotation
    xanglerad = float(xangle) / 180. * np.pi
    yanglerad = float(yangle) / 180. * np.pi
    zanglerad = float(zangle) / 180. * np.pi

    if zangle != 0: # Rotate about z
        rold = np.sqrt(x * x + y * y)
        phi = np.arctan2(y, x)
        phi -= zanglerad
        x = rold * np.cos(phi)
        y = rold * np.sin(phi)
    if yangle != 0: # Rotate about y
        rold = np.sqrt(z * z + x * x)
        phi = np.arctan2(z, x)
        phi -= yanglerad
        z = rold * np.sin(phi)
        x = rold * np.cos(phi)
    if xangle != 0: # Rotate about x
        rold = np.sqrt(y * y + z * z)
        phi = np.arctan2(z, y)
        phi -= xanglerad
        y = rold * np.cos(phi)
        z = rold * np.sin(phi)

    return x, y, z


def fix_axis_rotation(ax):
    def update(*args, **kwargs):
        if ax.azim < -360: ax.azim += 360
        elif ax.azim > 360: ax.azim -= 360
    ax.get_figure().canvas.mpl_connect('draw_event', update)

def plot_vertices(ax, xyz, origin=np.zeros(3), **kwargs):
    origin = np.asarray(origin)
    xyz = origin + np.asarray(xyz)
    if len(np.shape(xyz)) == 1 or np.shape(xyz)[1] <= 1:
        xyz = np.array([[xyz[0], xyz[1], xyz[2]]])
    
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], **kwargs)

    for vertex in xyz:
        ax.plot(
            [origin[0], vertex[0]],
            [origin[1], vertex[1]],
            [origin[2], vertex[2]],
            color='k',
            lw=0.5,
        )

    texts = []
    for vertex in xyz:
        radius = np.sqrt(sum((vertex - origin)**2))
        t = np.arccos(vertex[2] / radius)
        p = np.arctan2(vertex[1], vertex[0])
        zdir = tuple(vertex - origin)
        texts += [ax.text(
            vertex[0], vertex[1], vertex[2],
            '(%g, %g)' % (t, p),
            zdir,
            rotation_mode='anchor',
            ha='right',
        )]

    def update(*args, **kwargs):
        for i, vertex in enumerate(xyz):
            p = np.arctan2(vertex[1], vertex[0])
            angle = p / np.pi * 180. - ax.azim

            if angle > 360: angle -= 360
            elif angle < -360: angle += 360

            #print(texts[i].get_text(), angle)
            
            if ((angle >= 0 and angle < 180) or
                (angle < 0 and angle <= -180)):
                texts[i].set_ha('left')
            elif ((angle < 0 and angle > -180) or
                  (angle >= 0 and angle >= 180)):
                texts[i].set_ha('right')

        #print()
        ax.get_figure().canvas.draw_idle()

    ax.get_figure().canvas.mpl_connect('draw_event', update)


def draw_circle_on_sphere(ax, p:float, a:float, v:float, **kwargs):
    '''
        Parametric equation determined by the radius and angular positions (both polar and azimuthal relative to the z-axis) of the circle on the spherical surface
        Parameters:
            p (float): polar angle
            a (float): azimuthal angle
            v (float): radius is controlled by sin(v)
            
        Returns:
            Circular scatter points on a spherical surface
    '''
    
    u = np.mgrid[0:2*np.pi:30j]
    
    x = np.sin(v)*np.cos(p)*np.cos(a)*np.cos(u) + np.cos(v)*np.sin(p)*np.cos(a) - np.sin(v)*np.sin(a)*np.sin(u)
    y = np.sin(v)*np.cos(p)*np.sin(a)*np.cos(u) + np.cos(v)*np.sin(p)*np.sin(a) + np.sin(v)*np.cos(a)*np.sin(u)
    z = -np.sin(v)*np.sin(p)*np.cos(u) + np.cos(v)*np.cos(p)

    return ax.plot(x, y, z, **kwargs)
    

class TestIcoFluxFinder(basetest.BaseTest):
    def setUp(self):
        directory = os.path.join(
            starsmashertools.SOURCE_DIRECTORY,
            'tests',
            'flux_test',
        )
        self.simulation = starsmashertools.get_simulation(directory)

    def testAngles(self):
        # Making sure the angles we get from the log0.sph file are correct
        fluxfinder = starsmashertools.flux.icofluxfinder.IcoFluxFinder(
            self.simulation.get_output(),
        )
        
        for i, (t, p) in enumerate(zip(fluxfinder._theta, fluxfinder._phi)):
            self.assertAlmostEqual(t, angles[i][0])
            self.assertAlmostEqual(p, angles[i][1])

    def testGetClosestAngle(self):
        import copy
        
        fluxfinder = starsmashertools.flux.icofluxfinder.IcoFluxFinder(
            self.simulation.get_output(),
        )
        
        # Make a unit sphere
        fluxfinder.output['x'] = np.array([0.])
        fluxfinder.output['y'] = np.array([0.])
        fluxfinder.output['z'] = np.array([0.])
        fluxfinder.output['hp'] = np.array([0.5])
        radius = 2 * fluxfinder.output['hp'][0]
        vertices = radius * fluxfinder._unit_vertices

        # Test all the known angles
        for i, (t, p) in enumerate(angles):
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.sin(t) * np.sin(p)
            z = radius * np.cos(t)
            
            theta, phi = fluxfinder.get_closest_angle(x, y, z, 0)
            self.assertAlmostEqual(theta, t)
            self.assertAlmostEqual(phi, p)

        orig_vertices = copy.deepcopy(vertices)

        # Find the maximum displacement we can try
        mindr2 = np.inf
        for _t, _p in angles:
            tp = np.array([_t, _p])
            diff2 = np.sum((angles - tp)**2, axis=-1)
            mindr2 = min(
                mindr2,
                np.amin(diff2[diff2 != 0]),
            )
            

        mindist = np.sqrt(mindr2)
        eps = 1.e-7
        
        # Try a ring around each vertex
        Ntry = 50
        min_disp = 1 # In degrees
        max_disp = 0.5 * mindist * 180 / np.pi # In degrees
        
        dtheta = max_disp / 180. * np.pi
        dphis = np.linspace(min_disp, max_disp, Ntry) / 180. * np.pi

        orig_vertices = copy.deepcopy(fluxfinder._unit_vertices)

        _xyz_orig = radius * np.column_stack((
            np.sin(dtheta) * np.cos(dphis),
            np.sin(dtheta) * np.sin(dphis),
            np.full(len(dphis), np.cos(dtheta)),
        ))

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('equal')
        
        fix_axis_rotation(ax)

        # Make a ring around each angle
        for i, (_t, _p) in enumerate(angles):
            _x, _y, _z = rotate(
                _xyz_orig[:,0],
                _xyz_orig[:,1],
                _xyz_orig[:,2],
                xangle = 0.,
                yangle = _t / np.pi * 180.,
                zangle = _p / np.pi * 180.,
            )

            _xyz = np.column_stack((_x,_y,_z))

            """
            rotx, roty, rotz = rotate(
                copy.deepcopy(orig_vertices[:,0]),
                copy.deepcopy(orig_vertices[:,1]),
                copy.deepcopy(orig_vertices[:,2]),
                xangle = 0.,
                yangle = _t / np.pi * 180.,
                zangle = _p / np.pi * 180.,
            )

            fluxfinder._theta = np.arccos(rotz)
            fluxfinder._phi = np.arctan2(roty, rotx)
            fluxfinder._unit_vertices = np.column_stack((
                np.sin(fluxfinder._theta) * np.cos(fluxfinder._phi),
                np.sin(fluxfinder._theta) * np.sin(fluxfinder._phi),
                np.cos(fluxfinder._theta),
            ))
            """

            thet = np.arccos(_z / radius)
            ph = np.arctan2(_y, _x)
            
            for _x, _y, _z in _xyz:
                theta, phi = fluxfinder.get_closest_angle(_x, _y, _z, 0)

                diff2 = (thet - theta)**2 + (ph - phi)**2
                if diff2 > eps * eps:
                    message = "diff > eps: theta = %g, eps = %g" % (theta, eps)
                    print(message)
                    
                    idx = np.logical_and(
                        fluxfinder._theta == theta,
                        fluxfinder._phi == phi,
                    )
                    
                    plot_vertices(
                        ax,
                        radius * fluxfinder._unit_vertices[~idx],
                        color='b',
                    )
                    plot_vertices(
                        ax,
                        radius * fluxfinder._unit_vertices[idx],
                        color='g',
                        label = 'closest',
                    )
                    
                    plot_vertices(
                        ax,
                        np.array([_x, _y, _z]),
                        color='r',
                        label = 'position',
                    )

                    draw_circle_on_sphere(ax, 0., 0., dtheta, color='k', lw=0.5)

                    ax.legend()

                    plt.show()

                    raise AssertionError(message)


if __name__ == '__main__':
    import unittest
    unittest.main(failfast=True)
