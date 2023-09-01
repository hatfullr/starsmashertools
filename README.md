# starsmashertools

To install from source, download the code and run
```
./install
```

## Basic Usage

Using the tools requires a normal `StarSmasher` simulation directory. The minimum required files should be structured as follows:
```
simulation
   |- sph.input
   |- log*.sph
   |- out*.sph
   |- src
   | |- starsmasher.h
   | |- init.f
   | |- output.f
```
If the simulation is a dynamical simulation ('nrelax=0' in the sph.input file), then you may also need the file `restartrad.sph.orig`. You can modify the names which `starsmashertools` expects to find in a simulation directory in the `<starsmashertools directory>/starsmashertools/preferences.py` file. The source code directory is required to be present in each simulation directory, but its name is unimportant, as `starsmashertools` automatically detects the directory based on the `'src identifiers'` list in `preferences.py`.

If we assume that the directory tree above is a relaxation of a single star (nrelax=1), then we can start using `starsmashertools` in the following way:
```python
$ python3
Python 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import starsmashertools
>>> simulation = starsmashertools.get_simulation('simulation')
>>> simulation
<starsmashertools.lib.relaxation.Relaxation object at 0x7f96f323ace0>
>>> 
```
We can see that the directory has automatically been detected as a relaxation. From here we can access the data outputs directly
```python
>>> simulation = starsmashertools.get_simulation('simulation')
>>> output = simulation.get_output(0)
>>> output
Output('out0000.sph')
>>> output.simulation
<starsmashertools.lib.relaxation.Relaxation object at 0x7f4744a9e740>
>>> output.simulation.directory
'simulation'
>>> output['x']
array([ 1.00787165e-17, -3.23741675e+00, -3.23741667e+00, ...,
        3.30216492e+00,  3.30216494e+00,  3.30216505e+00])
>>> output['y']
array([ 8.25650447e-14, -5.23354366e-01, -5.98119466e-01, ...,
        1.86911995e-01,  2.61677505e-01,  1.86912292e-01])
>>> output['z']
array([-2.06446169e-14, -4.75801302e-01, -3.70067297e-01, ...,
        5.28667465e-02,  1.58600276e-01,  2.64334388e-01])
>>> 
```
An Output object is a python dictionary that reads its associated output file a single time on the first instance that data from it has been requested from it. You can obtain both particle data, such as 'x', 'y', and 'z', as well as header information, such as 'ntot'.

To retrieve large amounts of data at one time you can use an `OutputIterator` object:
```python
>>> iterator = simulation.get_output_iterator(start=0, stop=100, step=5)
>>> iterator
OutputIterator('out0000.sph' ... 'out0100.sph')
>>> for output in iterator:
...     print(output['t'])
...
0.007093005586200368
50.03439454661202
100.00578067955324
150.04551525866282
200.02984593162526
250.04367704538956
300.02365874121955
350.0439241833787
400.0364839590074
450.0012159900077
500.02571962973764
550.0198432855193
600.0417924218892
650.0236563963566
700.0472724485348
750.0377836388566
800.0140492636515
850.0236362175219
900.0143610996718
950.0412990029492
1000.0050735618534
>>> 
```
An `OutputIterator` is a powerful tool which by default reads the files asynchronously during iteration time. After an `OutputIterator` has finished iteration it cannot process those data files again, so a new iterator must be created.

Note that the outputs are in their raw form, or 'code units'. To retrieve values in cgs units you can set the mode:
```python
>>> output
Output('out0100.sph')
>>> output.mode = 'cgs'
>>> output['t']
1593633.8530935373
>>>
```

We also have access to other information about a simulation:
```python
>>> simulation['nrelax']
1
>>> simulation['tf']
1000
>>> simulation['ncooling']
0
```
This information comes both from the `sph.input` file and the `init.f` file in the source code directory. The `init.f` file is read first to set the default values and then those are overwritten by the values in `sph.input`.

We also have access to the simulations that may have preceeded the current one. For example, making a binary merger in `StarSmasher` starts with at least one stellar relaxation, which is then placed in a corotating frame in a binary scan, and then a dynamical simulation is created from the binary. To trace this pipeline in `starsmashertools` you can use the `get_children()` method:
```
>>> simulation = starsmashertools.get_simulation("/path/to/dynamical")
>>> children = simulation.get_children()
>>> children
[<starsmashertools.lib.binary.Binary object at 0x7f1543e0ec20>]
>>> children[0].get_children()
[<starsmashertools.lib.relaxation.Relaxation object at 0x7f15471670a0>, <starsmashertools.lib.relaxation.Relaxation object at 0x7f1543e0ee60>]
```
In order to use this functionality you need to inform `starsmashertools` about where to look for your simulations in the `preferences.py` file. The process of finding child simulations for the first time can take up to a few minutes depending on how many directories and subdirectories need to be searched. However, once a child is located it is stored in memory for instant access later, in `<starsmashertools directory>/data/children.json.gz`.


## Making Plots

`starsmashertools` has integrated Matplotlib functionality to help streamline the plotting of data. It is very common to have a simulation directory which contains perhaps hundreds of GB worth of data such that reading and processing all the data can take a long time. If your plotting code has an error in it after reading all that data then you have to start the whole process again. `starsmashertools` combats this problem with the `PlotData` object which creates a checkpoint file during data processing.
```python
import starsmashertools
import starsmashertools.mpl.plotdata
import matplotlib.pyplot as plt
import numpy as np

# Get the total mass of ejected particles
def get_Mejecta(output):
    idx = output['unbound'] # 'unbound' is defined in preferences.py
    if np.any(idx):
        return np.sum(output['am'][idx])
    return 0.

# Get the time in days
def get_time(output):
    output.mode = 'cgs'
    return output['t'] / (3600. * 24.)

simulation = starsmashertools.get_simulation('simulation')

time, Mejecta = starsmashertools.mpl.plotdata.PlotData(
    simulation.get_outputfiles(),
    [get_time, get_Mejecta],
    simulation = simulation,
    checkpointfile = 'pdc.json.gz', # The default name ('pdc' = 'PlotData Checkpoint')
    read = True, # Read pdc.json.gz if it already exists
    write = True, # Overwrite the pdc.json.gz if it exists
    # You can also specify instead of 'checkpointfile' separate files for
    # reading and writing with the 'readfiles' and 'writefile' keywords.
    # 'readfiles' can be either a string or an iterable of strings.
    #readfiles = ['pdc0.json.gz', 'pdc1.json.gz'],
    #writefile = 'mypdc.json.gz',
)

plt.plot(time, Mejecta)

plt.show()
```
You may notice that the checkpoint file is created only on every 100 output files read. You can modify this behavior in the `preferences.py` file under `OutputIterator`, `max buffer size`, or you can set it manually in your code by passing in an `OutputIterator` to the `PlotData` object instead of a list of filenames:
```
iterator = simulation.get_output_iterator(max_buffer_size=10)
time, Mejecta = starsmashertools.mpl.plotdata.PlotData(
    iterator,
    [get_time, get_Mejecta],
    simulation = simulation,
    checkpointfile = 'pdc.json.gz', # The default name ('pdc' = 'PlotData Checkpoint')
    read = True, # Read pdc.json.gz if it already exists
    write = True, # Overwrite the pdc.json.gz if it exists
)
```