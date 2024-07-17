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

If you don't have an internet connection, you can try the offline installer:

.. code-block:: console

   ./install_offline


Simulation Directories
----------------------

To use ``starsmashertools``, your ``StarSmasher`` simulations must be structured at a bare minimum in the following way:

.. code-block::

   |-- simulation-directory
   |   |-- sph.input
   |   |-- src
   |       |-- init.f
   |       |-- output.f
   |       |-- starsmasher.h

Other files may also be required depending on if the simulation is a stellar relaxation (``nrelax=1``), a binary scanning run (``nrelax=2`` or ``nrelax=3``), or a dynamical calculation (``nrelax=0``). Here is an example directory structure of a generic ``StarSmasher`` simulation that has been run and produced 5 output files:

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
   
    

Using the API
-------------

Here is an example run from a simulation directory, which happens to be a dynamical simulation:

.. code-block:: python

   >>> import starsmashertools
   >>> simulation = starsmashertools.get_simulation('.')
   >>> simulation
   <starsmashertools.lib.dynamical.Dynamical object at 0x7f4744a9e740>
   >>> simulation['nrelax']
   0

   
Let's inspect the simulation's output files:
	     
.. code-block:: python

   >>> simulation.get_output()
   [Output('out0000.sph'), Output('out0001.sph'), Output('out0002.sph'), Output('out0003.sph'), Output('out0004.sph')]

   
An :class:`~starsmashertools.lib.output.Output` object functions like a :py:class:`dict` except information is read from the files only after the first :meth:`~starsmashertools.lib.output.Output.__getitem__` request, after which the dictionary is filled.

.. code-block:: python

   >>> first = simulation.get_output(0)
   >>> first['x']
   array([ 1.00787165e-17, -3.23741675e+00, -3.23741667e+00, ...,
        3.30216492e+00,  3.30216494e+00,  3.30216505e+00])


Header information from the output files is also available, as well as special additional cached data which can be edited and ammended in your preferences file ``starsmashertools/data/user/preferences.py``\. If you do not have a preferences file yet, copy the one from ``starsmashertools/data/defaults/preferences.py``\.

If your use case involves long calculations, you can defer the reading of output files to a separate process to speed up your code:

.. code-block:: python

   >>> simulation.get_output_iterator()
   OutputIterator('out0000.sph' ... 'out0004.sph')

See the :class:`~starsmashertools.lib.output.OutputIterator` and :class:`~starsmashertools.lib.output.Output` classes for more details.


Using the CLI
-------------

There are some Command Line Interface (CLI) programs included with ``starsmashertools``. Inside a simulation directory, run the ``starsmashertools`` command, which will launch a CLI application that looks like:

.. code-block:: console

     Main Menu

     Directory = [snipped]

     Choose an option
          0) Set children (set_children)
          1) Show children (get_children)
          2) Show output files (get_output)
          3) Plot energies (plot_energy)
          4) Plot animation (plot_animation)
          5) Concatenate simulations (join)
          6) Detach concatenated simulations (split)
          7) Show concatenated simulations (show_joined_simulations)
          8) get_relaxations
          9) get_binary
         10) get_plunge_time

    q) quit
    : 

Sometimes the CLI works well, and sometimes it doesn't. Sorry about that. It's always safe to fallback to using iPython in the terminal.
