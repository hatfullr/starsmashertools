import starsmashertools.lib.fluxfinder


class IcoFluxFinder(starsmashertools.lib.fluxfinder.FluxFinder, object):
    """
    A type of FluxFinder which uses the "subdivided icosahedron" angles that are
    used in the modified version of StarSmasher written by Hatfull et al. (2024;
    in progress).
    """
    def __init__(self, *args, **kwargs):
        super(IcoFluxFinder, self).__init__(*args, **kwargs)

        logfiles = self.output.simulation.get_logfiles()
        if logfiles is None:
            #
            pass
