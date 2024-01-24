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
        import copy
        directory = os.path.join(
            starsmashertools.SOURCE_DIRECTORY,
            'tests',
            'flux_test',
        )
        self.simulation = starsmashertools.get_simulation(directory)

        self.output = copy.deepcopy(self.simulation.get_output())
        # Make a unit sphere
        self.output['x'] = np.zeros(1)
        self.output['y'] = np.zeros(1)
        self.output['z'] = np.zeros(1)
        self.output['hp'] = np.ones(1) * 0.5
        radius = 2 * self.output['hp'][0]
        self.output['am'] = np.ones(1)
        self.output['u'] = np.ones(1)
        self.output['rho'] = self.output['am'] / (4./3. * np.pi * (2 * self.output['hp'])**3)
        self.output['tau'] = np.ones(1) * 1.e30
        self.output['popacity'] = np.ones(1)
        
        for key, val in self.output.items():
            if isinstance(val, np.ndarray):
                if key not in ['x','y','z','hp','am','u','rho','tau','popacity']:
                    self.output[key] = np.zeros(1)

        self.output['dEdiffdt'] = np.zeros(1)
        self.output['uraddot'] = np.zeros(1)
                    
        self.output['ntot'] = 1

    def testAngles(self):
        # Making sure the angles we get from the log0.sph file are correct
        fluxfinder = starsmashertools.flux.icofluxfinder.IcoFluxFinder(
            self.output,
        )
        
        for i, (t, p) in enumerate(zip(fluxfinder._theta, fluxfinder._phi)):
            self.assertAlmostEqual(t, angles[i][0])
            self.assertAlmostEqual(p, angles[i][1])

    def testUnitVertices(self):
        fluxfinder = starsmashertools.flux.icofluxfinder.IcoFluxFinder(
            self.output,
        )

        expected_vertices = np.column_stack((
            fluxfinder.output['radius'] * np.sin(angles[:,0]) * np.cos(angles[:,1]),
            fluxfinder.output['radius'] * np.sin(angles[:,0]) * np.sin(angles[:,1]),
            fluxfinder.output['radius'] * np.cos(angles[:,0]),
        ))
        
        for expected, vertex in zip(expected_vertices, fluxfinder._unit_vertices):
            self.assertAlmostEqual(expected[0], vertex[0], msg = "Vertex mismatch on x-axis: (%g, %g, %g) != (%g, %g, %g)" % (*expected, *vertex))
            self.assertAlmostEqual(expected[1], vertex[1], msg = "Vertex mismatch on y-axis: (%g, %g, %g) != (%g, %g, %g)" % (*expected, *vertex))
            self.assertAlmostEqual(expected[2], vertex[2], msg = "Vertex mismatch on z-axis: (%g, %g, %g) != (%g, %g, %g)" % (*expected, *vertex))
        

    def testGetClosestAngle(self):
        import copy
        import starsmashertools.math
        
        fluxfinder = starsmashertools.flux.icofluxfinder.IcoFluxFinder(
            self.output,
        )
        
        vertices = fluxfinder.output['radius'] * fluxfinder._unit_vertices

        """
        xyz = np.column_stack((
            fluxfinder.output['radius'] * np.sin(angles[:,0]) * np.cos(angles[:,1]),
            fluxfinder.output['radius'] * np.sin(angles[:,0]) * np.sin(angles[:,1]),
            fluxfinder.output['radius'] * np.cos(angles[:,0]),
        ))
        
        # Test all the known angles
        for i, (t, p) in enumerate(angles):
            theta, phi = fluxfinder.get_closest_angle(xyz[i], 0)
            self.assertAlmostEqual(theta, t, msg="theta problem: closest angle to (%g, %g) was found as not that angle, but rather (%g, %g)" % (t, p, theta, phi))
            self.assertAlmostEqual(phi, p, msg="phi problem: closest angle to (%g, %g) was found as not that angle, but rather (%g, %g)" % (t, p, theta, phi))
        """

        Ntry = 50
        eps = 1.e-7
        
        # Get the minimum angular displacement each vertex has from each other
        # vertex
        min_angular_separation = np.inf
        v = vertices[0]
        v_norm = v #/ np.sqrt(np.sum(v**2, axis=-1))
        for vertex in vertices[1:]:
            #norm = vertex / np.sqrt(np.sum(vertex**2, axis=-1))
            #costheta = np.dot(v_norm, norm)
            costheta = np.dot(v, vertex)
            theta = np.arccos(costheta)
            min_angular_separation = min(min_angular_separation, theta)

        min_angular_separation -= eps
            
        # For each vertex, we will walk along the surface of the sphere to test
        # if the closest angle always corresponds to that vertex. We stay within
        # a maximum angular displacement of half the minimum angular
        # displacement between any two vertices.

        thetas = np.linspace(eps, 0.5 * min_angular_separation, Ntry)
        phis = np.linspace(0., 2*np.pi, Ntry + 1)[:-1]
        
        trial_positions = []
        for theta in thetas:
            sintheta = np.sin(theta)
            costheta = np.cos(theta)
            for phi in phis:
                trial_positions += [[
                    sintheta * np.cos(phi),
                    sintheta * np.sin(phi),
                    costheta,
                ]]
        trial_positions = np.asarray(trial_positions)
        trial_positions *= fluxfinder.output['radius']
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = plt.axes(projection='3d', computed_zorder=False)
        ax.set_aspect('equal')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        fix_axis_rotation(ax)

        plot_vertices(
            ax,
            fluxfinder.output['radius'] * fluxfinder._unit_vertices,
            color = 'b',
            label = 'unit vertices',
            zorder = 0,
        )
        
        for i, (_t, _p) in enumerate(angles):
            attempt = fluxfinder.output['radius'] * np.column_stack((
                np.sin(_t) * np.cos(_p),
                np.sin(_t) * np.sin(_p),
                np.cos(_t),
            ))
            
            _trial_x, _trial_y, _trial_z = starsmashertools.math.rotate_spherical(
                copy.deepcopy(trial_positions[:,0]),
                copy.deepcopy(trial_positions[:,1]),
                copy.deepcopy(trial_positions[:,2]),
                theta = _t / np.pi * 180,
                phi = _p / np.pi * 180,
            )
            
            _trial_positions = np.column_stack((_trial_x, _trial_y, _trial_z))
            
            for p in _trial_positions:
                theta, phi = fluxfinder.get_closest_angle(p, 0)
                diff = np.sqrt((theta - _t)**2 + (phi - _p)**2)
                if diff > eps:
                    message = "diff > 2*eps: diff = %g, eps = %g, closest = (%g, %g), position = (%g, %g)" % (diff, eps, theta, phi, _t, _p)
                    #print(xyzfound, p)
                    print(message)

                    unit_vertices = fluxfinder.output['radius'] * fluxfinder._unit_vertices
                    other_vertices = unit_vertices[:i].tolist() + unit_vertices[i+1:].tolist()
                    other_vertices = np.asarray(other_vertices)
                    
                    ax.scatter(
                        attempt[:,0], attempt[:,1], attempt[:,2],
                        color = 'm',
                        label = 'testing vertex',
                        zorder = 1000,
                    )

                    ax.scatter(
                        fluxfinder.output['radius'] * np.sin(theta) * np.cos(phi),
                        fluxfinder.output['radius'] * np.sin(theta) * np.sin(phi),
                        fluxfinder.output['radius'] * np.cos(theta),
                        color = (0., 1., 0.),
                        label = 'found closest angle',
                        zorder = 100000,
                    )

                    plot_vertices(
                        ax,
                        p,
                        color='r',
                        label = 'test position',
                        zorder=900,
                    )
                    
                    ax.scatter(
                        _trial_positions[:,0],
                        _trial_positions[:,1],
                        _trial_positions[:,2],
                        color='k',
                        s=0.01,
                    )
                    
                    circle = draw_circle_on_sphere(
                        ax, 0., 0., 0.5 * min_angular_separation,
                        color='k', lw=0.5,
                    )[0]
                    cx, cy, cz = circle.get_data_3d()

                    cx, cy, cz = starsmashertools.math.rotate_spherical(
                        cx, cy, cz,
                        theta = _t / np.pi * 180,
                        phi = _p / np.pi * 180,
                    )
                    circle.set_data_3d(cx, cy, cz)
                    
                    ax.legend()

                    plt.show()

                    raise AssertionError(message)


if __name__ == '__main__':
    import unittest
    unittest.main(failfast=True)
