import starsmashertools.preferences
from starsmashertools.preferences import Pref
import typing

def get_locks(index : int):
    import starsmashertools.helpers.path
    from starsmashertools import LOCK_DIRECTORY
    for entry in starsmashertools.helpers.path.listdir(LOCK_DIRECTORY):
        if entry == 'GPU' + str(index):
            yield starsmashertools.helpers.path.join(LOCK_DIRECTORY, entry)

def is_device_available(index : int):
    r"""
    Returns ``True`` if there are no :class:`~.GPUJob` objects currently running
    on the device of index ``index``\.

    Parameters
    ----------
    index : int
        The device index.

    Returns
    -------
    bool
        ``True`` if the device is available, ``False`` otherwise.
    """
    import starsmashertools.helpers.path
    from starsmashertools import LOCK_DIRECTORY
    path = starsmashertools.helpers.path.join(LOCK_DIRECTORY, 'GPU' + str(index))
    return not starsmashertools.helpers.path.exists(path)


try:
    import starsmashertools.helpers.argumentenforcer
    import starsmashertools.lib.output
    import time
    import numpy as np

    import numba
    from numba import cuda

    import warnings
    import math

    @starsmashertools.preferences.use
    class GPUJob(object):
        r"""
        A base class for utilizing the GPU.
        """
        def __new__(cls, *args, **kwargs):
            instance = super(GPUJob, cls).__new__(cls)
            instance._outputs = []
            return instance

        def __init__(
                self,
                inputs : list | tuple = [],
                outputs : list | tuple = [],
                kernel : typing.Callable | type(None) = None,
        ):
            self.inputs = inputs
            if outputs: self.outputs = outputs
            if hasattr(self, 'kernel') and kernel is not None:
                raise ValueError("Argument 'kernel' cannot be None when the implementing class of a GPUJob already implements a function called 'kernel'")
            if not hasattr(self, 'kernel'):
                if kernel is None:
                    raise ValueError("Argument 'kernel' must have a value when the implementing class of a GPUJob does not implement a function called 'kernel'")
                self.kernel = kernel
            self.device = None

        @property
        def outputs(self): return self._outputs
        @outputs.setter
        def outputs(self, value):
            if len(value) == 0:
                raise ValueError("Invalid output arguments: '%s'" % str(value))

            self._outputs = value
            self._resolution = None
            for i in range(len(self._outputs)):
                if not isinstance(self._outputs[i], np.ndarray):
                    self._outputs[i] = np.asarray(self._outputs[i])
                if self._resolution is None: self._resolution = self._outputs[i].shape
                elif self._outputs[i].shape != self._resolution:
                    raise Exception("Outputs must all have the same shape")

        def get_device(self):
            r"""
            Block until there exists a GPU without a GPUJob running on it. When
            a device becomes available, record in data/locks that the GPU is
            being used.
            """
            import starsmashertools.helpers.file
            from starsmashertools import LOCK_DIRECTORY
            import starsmashertools.helpers.path
            import time
            
            if self.device is not None:
                raise Exception("The GPUJob has already device. Call release_device() first.")
            
            # Wait for an available GPU
            self.device = -1
            while self.device == -1:
                for i, device in enumerate(cuda.gpus):
                    if not is_device_available(i): continue
                    cuda.select_device(i)
                    self.device = i
                    break
                time.sleep(1.e-2)
            # Lock the GPU
            with open(
                    starsmashertools.helpers.path.join(
                        LOCK_DIRECTORY, 'GPU' + str(self.device),
                    ),
                    'x',
            ) as f:
                f.write(' ')
        

        def release_device(self):
            import starsmashertools.helpers.path
            path = starsmashertools.helpers.path.join(
                LOCK_DIRECTORY, 'GPU' + str(device_index)
            )
            if starsmashertools.helpers.path.exists(path):
                starsmashertools.helpers.path.remove(path)
            self.device = None
            cuda.close()
            
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def run(
                self,
                threadsperblock : int = Pref('run.threadsperblock'),
                return_duration : bool = False,
        ):
            import starsmashertools
            
            self.get_device()

            try:
                # Send inputs to the GPU
                inputs = []
                for i in range(len(self.inputs)):
                    if hasattr(self.inputs[i], '__iter__') and not isinstance(self.inputs[i], str):
                        inputs += [cuda.to_device(np.ascontiguousarray(self.inputs[i]))]
                    else:
                        inputs += [self.inputs[i]]

                # Send outputs to the GPU
                outputs = []
                for i in range(len(self.outputs)):
                    outputs += [cuda.to_device(np.ascontiguousarray(self.outputs[i]))]

                # Determine allocation sizes on the GPU
                if len(self._resolution) == 1:
                    blockspergrid = self._resolution[0] // threadsperblock + 1
                else:
                    ndims = len(self._resolution)
                    threadsperblock = np.full(ndims, int(threadsperblock**(1./ndims)), dtype=int)

                    blockspergrid = np.array(self._resolution, dtype=int) // threadsperblock + 1
                    threadsperblock = tuple(threadsperblock)
                    blockspergrid = tuple(blockspergrid)

                start_time = time.time()

                # These warnings are caused by under-utilizing the device, which we
                # typically don't have any control over
                warnings.filterwarnings(
                    action='ignore',
                    category=numba.core.errors.NumbaPerformanceWarning,
                )
                self.kernel[blockspergrid, threadsperblock](
                    *outputs,
                    *inputs,
                )
                warnings.resetwarnings()
                cuda.synchronize()

                for i in range(len(outputs)):
                    outputs[i] = outputs[i].copy_to_host()

                if len(outputs) == 1: outputs = outputs[0]
            except Exception as e:
                self.release_device()
                raise e

            self.release_device()
            
            if return_duration:
                return outputs, time.time() - start_time
            return outputs












    class GravitationalPotentialEnergies(GPUJob, object):
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def __init__(
                self,
                output : starsmashertools.lib.output.Output,
        ):
            r"""
            Given a `~.Output` object, calculate the gravitational potential
            energies of each particle.
            """

            if output.simulation['nkernel'] not in [0, 1, 2]:
                raise NotImplementedError("nkernel '%s' is not supported" % str(output.simulation['nkernel']))

            super(GravitationalPotentialEnergies, self).__init__()

            units = output.simulation.units
            ntot = len(output['x'])
            self.inputs = [
                output['am'],
                output['x'],
                output['y'],
                output['z'],
                output['hp'],
                ntot,
                output.simulation['nkernel'],
            ]
            self.outputs = [
                np.full(ntot, np.nan),
            ]


        @staticmethod
        @cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32, int32)')
        def kernel(result, mass, x, y, z, hp, N, nkernel):
            i = cuda.grid(1)
            if i < N:
                ami = mass[i]
                xi = x[i]
                yi = y[i]
                zi = z[i]
                hpi = hp[i]
                twohpi = 2*hpi
                fourhpi2 = twohpi*twohpi
                invhpi = 1. / twohpi
                invhpi2 = invhpi * invhpi

                result[i] = 0.
                for j in range(N):
                    hpj = hp[j]
                    twohpj = 2*hpj
                    fourhpj2 = twohpj * twohpj
                    amj = mass[j]

                    dx = xi - x[j]
                    dy = yi - y[j]
                    dz = zi - z[j]
                    r2 = dx * dx + dy * dy + dz * dz

                    rinv = 1. / math.sqrt(r2)
                    mrinv1 = rinv * amj
                    gpot = 0
                    if r2 >= max(fourhpi2, fourhpj2):
                        gpot = -mrinv1
                    else:
                        # This comes from grav_force_direct.cu
                        rinv2 = rinv * rinv
                        mrinv3 = rinv2 * mrinv1

                        invhpj = 1./twohpj
                        invhpj2 = invhpj * invhpj

                        invqx, invqy = 0, 0
                        qx, qy = 0, 0
                        if r2 > 0:
                            invqx, invqy = rinv * twohpi, rinv * twohpj
                            qx, qy = 1./invqx, 1./invqy
                        q2x, q2y = qx*qx, qy*qy

                        mj1x, mj1y = amj * invhpi, amj * invhpj
                        mj2x, mj2y = mj1x * invhpi2, mj1y * invhpj2
                        hflag = 1 if hpj > 0 else 0

                        gx, gy = int(r2 <= fourhpi2), int(r2 <= fourhpj2)

                        if nkernel == 0:
                            fx, fy = 1 if qx < 0.5 else 0, 1 if qy < 0.5 else 0
                            if r2 <= 0:
                                gpot = -1.4*hflag*(mj1x + mj1y)
                            else:
                                potx = fx * (-2.8 + q2x * (5.333333333333  + q2x * (6.4 * qx - 9.6))) + \
                                    (1 - fx) * (-3.2 + 0.066666666667 * invqx + q2x * (10.666666666667 + qx * (-16 + qx * (9.6 - 2.133333333333 * qx))))

                                poty = fy * (-2.8 + q2y * (5.333333333333 + q2y * (6.4 * qy - 9.6))) + \
                                    (1 - fy) * (-3.2 + 0.066666666667 * invqy + q2y * (10.666666666667 + qy * (-16 + qy * (9.6 + -2.133333333333 * qy))))

                                if r2 < min(fourhpi2, fourhpj2):
                                    gpot = 0.5*(gx*mj1x*potx + (gx - 1)*mrinv1 + \
                                                 gy*mj1y*poty + (gy - 1)*mrinv1)
                                elif r2 < fourhpi2:
                                    gpot = 0.5*(-mrinv1 + mj1x*potx)
                                else:
                                    gpot = 0.5*(-mrinv1 + mj1y*poty)

                        elif nkernel == 1:
                            if r2 <= 0:
                                gpot = -1.9140625*hflag*(mj1x + mj1y)
                            else:
                                potx = -3.828125 + q2x * (14.21875 + q2x * (-46.921875 + q2x * (134.0625 + q2x * (-547.421875 + qx * (1001 + qx * ( -895.78125 + qx * (455 + qx * (-126.328125 + 15*qx))))))))
                                poty = -3.828125 + q2y * (14.21875 + q2y * (-46.921875 + q2y * (134.0625 + q2y * (-547.421875 + qy * (1001 + qy * ( -895.78125 + qy * (455 + qy * (-126.328125 + 15*qy))))))))
                                if r2 < min(fourhpi2, fourhpj2):
                                    gpot = 0.5*(mj1x*potx + (gx - 1)*mrinv1 + \
                                                mj1y*poty + (gy - 1)*mrinv1)
                                elif r2 < fourhpi2:
                                    gpot = 0.5*(mj1x*potx - mrinv1)
                                else:
                                    gpot = 0.5*(mj1y*poty - mrinv1)
                        elif nkernel == 2:
                            if r2 <= 0:
                                gpot = -1.71875*hflag*(mj1x + mj1y)
                            else:
                                potx = -3.4375 + q2x * (10.3125 + q2x * ( -28.875  + q2x * (103.125 + qx *(-165 + qx * (120.3125 + qx * (-44 + 6.5625*qx))))))
                                poty = -3.4375 + q2y * (10.3125 + q2y * ( -28.875  + q2y * (103.125 + qy *(-165 + qy * (120.3125 + qy * (-44 + 6.5625*qy))))))
                                if r2 < min(fourhpi2, fourhpj2):
                                    gpot = 0.5*(mj1x*potx + mj1y*poty)
                                elif r2 < fourhpi2:
                                    gpot = 0.5*(mj1x*potx - mrinv1)
                                else:
                                    gpot = 0.5*(mj1y*poty - mrinv1)
                        else:
                            gpot = np.nan
                    result[i] += gpot




# Catch this pesky exception which likes to pop up in random places
except Exception as e:# (numba.cuda.cudadrv.driver.CudaAPIError, numba.cuda.cudadrv.error.CudaSupportError) as e:
    if ('Call to cuInit results in UNKNOWN_CUDA_ERROR (804)' in str(e) or
        '[804] Call to cuInit results in UNKNOWN_CUDA_ERROR' in str(e)):
        raise RuntimeError("This error is known to happen for computers that have been put in 'suspend' mode some time after a restart. You can try logging out and logging back in again or restarting your system.") from e
    raise
