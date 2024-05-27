Getting Started
===============

.. _installation:

Installation
------------

First clone the GitHub repository and enter the directory:

.. code-block:: console

   git clone https://github.com/hatfullr/starsmashertools
   cd starsmashertools

Then run the install script:

.. code-block:: console

   ./install

If you don't have an internet connection (how did you get here?) you can use the offline installer:

.. code-block:: console

   ./install_offline


Simulation Directories
----------------------

To use `starsmashertools`, your `StarSmasher` simulations must be structured at a bare minimum in the following way:

.. code-block::

   |-- simulation-directory
   |   |-- sph.input
   |   |-- src
   |       |-- init.f
   |       |-- output.f
   |       |-- starsmasher.h

Other files may also be required depending on if the simulation is a stellar relaxation (`nrelax=1`), a binary scanning run (`nrelax=2` or `nrelax=3`), or a dynamical calculation (`nrelax=0`). Here is an example directory structure of a generic `StarSmasher` simulation that has been run and produced 5 output files:

.. code-block::
   :caption: Simulation directory structure

   |-- simulation-directory
   |   |-- energy0.sph
   |   |-- log0.sph
   |   |-- out0000.sph
   |   |-- out0001.sph
   |   |-- out0002.sph
   |   |-- out0003.sph
   |   |-- out0004.sph
   |   |-- simulation-directory_gpu_sph
   |   |-- sph.init
   |   |-- sph.input
   |   |-- src
   |       |-- init.f
   |       |-- output.f
   |       |-- starsmasher.h
   |       |-- ...
   

Using the CLI
-------------

The Command Line Interface (CLI) programs included with `starsmashertools` make analyzing `StarSmasher` simulation a breeze. Inside the simulation directory we run the following command:

.. code-block:: console

   starsmashertools

This will launch a CLI application which might look something like this:

.. code-block:: console

    Main Menu

    Directory = .../simulation-directory

    Choose an option
         0) get_children
         1) get_n
         2) get_final_radius

   q) quit
   : 


Using the API
-------------

For more granular control you can use `starsmashertools` as a module in any Python3 script:

.. code-block:: python
   :caption: Example API usage to get a list containing Output objects

   import starsmashertools
   simulation = starsmashertools.get_simulation('simulation-directory')
   print(simulation)
   print(simulation['nrelax'])

.. code-block :: console
   :caption: Console output
	     
   <starsmashertools.lib.dynamical.Dynamical object at 0x7f4744a9e740>
   0

Here `starsmashertools` has automatically detected the simulation as a dynamical calculation, and we can see that the `nrelax` input variable is `0` as expected. Inspecting the simulation outputs is simple:
	     
.. code-block:: python
   :caption: Simulation outputs

   print(simulation.get_output())

.. code-block:: console
   :caption: Console output

   [Output('out0000.sph'), Output('out0001.sph'), Output('out0002.sph'), Output('out0003.sph'), Output('out0004.sph')]

The output files in the simulation directory are stored as :py:meth:`~starsmashertools.lib.output.Output` objects, which function like Python dictionaries except their values cannot be modified and information is read from the files only after the first `__getitem__` request, such as `output['x']` to get the particle x positions:

.. code-block:: python

   first = simulation.get_output(0)
   print(first['x'])

.. code-block :: console
   :caption: Console output

   array([ 1.00787165e-17, -3.23741675e+00, -3.23741667e+00, ...,
        3.30216492e+00,  3.30216494e+00,  3.30216505e+00])

Header information from the output files is also available, as well as special additional cached data which can be edited and ammended in the `starsmashertools.preferences` file located in `starsmashertools/starsmashertools/preferences.py` under `'Output'` in `'cache'`. A list of available keys can be found using the `keys()` method on an :py:meth:`~starsmashertools.lib.output.Output` object.

For accelerated file reading when dealing with many output files we suggest you use an :py:meth:`~starsmashertools.lib.output.OutputIterator`:

.. code-block:: python
   :caption: Example API usage to get an OutputIterator

   import starsmashertools
   simulation = starsmashertools.get_simulation('simulation-directory')
   print(simulation.get_output_iterator())

.. code-block:: console
   :caption: Console output
   
   OutputIterator('out0000.sph' ... 'out0004.sph')

An OutputIterator "reads ahead" asynchronously to prepare output objects in the background while your code runs, which can be helpful for speeding up slow analysis tasks. See the :py:meth:`~starsmashertools.lib.output.OutputIterator` and :py:meth:`~starsmashertools.lib.output.Output` classes for more details.
