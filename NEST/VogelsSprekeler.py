#!/usr/bin/env python3
"""
NEST implementation of Vogels-Sprekeler network model.

File: VogelsSprekeler.py

*** NOTE: Requires NEST >= 2.12 ***
https://github.com/nest/nest-simulator/releases/tag/v2.12.0

Copyright 2018 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import nest
# use random.sample instead of numpy.random - faster
import random
import numpy
from mpi4py import MPI
import logging


class VogelsSprekeler:

    """VogelsSprekeler 2016"""

    def __init__(self):
        """Initialise variables."""
        self.comm = MPI.COMM_WORLD
        # default resolution in nest is 0.1ms. Using the same value
        # http://www.nest-simulator.org/scheduling-and-simulation-flow/
        self.dt = 0.1

        # populations
        self.populations = {'E': 8000, 'I': 2000, 'STIM': 1000, 'Poisson': 1,
                            'P': 784, 'R': 196}

        # Define number of columns for grid
        # Excitatory neurons
        self.colsE = 80
        # Inhibitory neuron
        self.colsI = 20
        # For the grid
        self.locations = {}
        # for the three patterns and stimulus to one pattern
        self.pattern1 = []
        self.pattern2 = []
        self.pattern3 = []
        self.recall1 = []

        # time recall stimulus is enabled for
        self.recall_duration = 1000.  # ms

        self.rank = nest.Rank()

        # Supplementary material documents how this affects the network
        self.wbar = 3.0
        self.weightEE = self.wbar
        self.weightII = self.wbar * -10.
        self.weightEI = self.wbar
        self.weightPatternEE = self.wbar * 5.
        self.weightExt = 50.

        random.seed(42)

    def __setup_neurons(self):
        """Setup properties of neurons."""
        # see the aif source for symbol definitions
        self.neuronDict = {'V_m': -60.,
                           't_ref': 5.0, 'V_reset': -60.,
                           'V_th': -50., 'C_m': 200.,
                           'E_L': -60., 'g_L': 10.,
                           'E_ex': 0., 'E_in': -80.,
                           'tau_syn_ex': 5., 'tau_syn_in': 10.,
                           }
        # Set up TIF neurons
        # Excitatory neuron model
        nest.CopyModel("iaf_cond_exp", "tif_neuronE")
        nest.SetDefaults("tif_neuronE", self.neuronDict)
        # Inhibitory neuron model
        nest.CopyModel("iaf_cond_exp", "tif_neuronI")
        nest.SetDefaults("tif_neuronI", self.neuronDict)

        # external stimulus
        self.poissonExtDict = {'rate': 10., 'origin': 0., 'start': 0.}

    def __create_neurons(self):
        """Create our neurons."""
        self.neuronsE = nest.Create('tif_neuronE', self.populations['E'])
        self.neuronsI = nest.Create('tif_neuronI', self.populations['I'])

        # Excitatory neurons
        if self.rank == 0:
            loc_file = open("00-neuron-locations-E.txt", 'w')
        for neuron in self.neuronsE:
            row = int((neuron - self.neuronsE[0])/self.colsE)
            col = ((neuron - self.neuronsE[0]) % self.colsE)
            self.locations[neuron] = numpy.array([col, row])
            if self.rank == 0:
                print("{}\t{}\t{}".format(neuron, col, row), file=loc_file)
        if self.rank == 0:
            loc_file.close()

        # Inhibitory neurons are spread evenly among excitatory neurons
        if self.rank == 0:
            loc_file = open("00-neuron-locations-I.txt", 'w')
        for neuron in self.neuronsI:
            row = int((neuron - self.neuronsI[0])/self.colsI)
            col = self.colsE + ((neuron - self.neuronsI[0]) % self.colsI)
            self.locations[neuron] = numpy.array([col, row])
            if self.rank == 0:
                print("{}\t{}\t{}".format(neuron, col, row),
                      file=loc_file)
        if self.rank == 0:
            loc_file.close()

        # External stimulus consists of poisson neurons (neurons that provide
        # spikes with an ISI taken from a poisson distribution). See NEST
        # documentation for details.
        self.poissonExt = nest.Create('poisson_generator',
                                      self.populations['Poisson'],
                                      params=self.poissonExtDict)

    def __setup_initial_connection_params(self):
        """Setup connections."""
        # Global sparsity
        self.sparsity = 0.02

        # Stimulus used when patterns are recalled
        self.sparsityStim = 0.05
        self.connectionNumberStim = int((self.populations['STIM'] *
                                         self.populations['R']) *
                                        self.sparsityStim)
        self.connDictStim = {'rule': 'fixed_total_number',
                             'N': self.connectionNumberStim}

        # each neuron gets a single spike train as external input
        self.connDictExt = {'rule': 'fixed_indegree',
                            'indegree': 1}

        # A weight of 1 translates to 1nS
        # A negative weight implies inhibitory conductance
        # Excitatory synapses (all static: EE, EI)
        nest.CopyModel('static_synapse', 'static_synapse_ex')
        # Inhibitory synapses (static: II)
        nest.CopyModel('static_synapse', 'static_synapse_in')
        # Inhibitory synapses (plastic with VogelsSprekeler STDP: IE)
        nest.CopyModel('vogels_sprekeler_synapse', 'stdp_synapse_in')

        # Set up synapse parameter dictionaries
        self.synDictEE = {'model': 'static_synapse_ex',
                          'weight': self.weightEE}
        self.synDictEI = {'model': 'static_synapse_ex',
                          'weight': self.weightEI}
        self.synDictII = {'model': 'static_synapse_in',
                          'weight': self.weightII}
        self.synDictIE = {'model': 'stdp_synapse_in',
                          'weight': -0.0000001, 'Wmax': -30000.,
                          'alpha': .12, 'eta': 0.01,
                          'tau': 20.}

    def __create_network_connections(self):
        """Connect various neuron sets."""
        # Connect external input to E neurons
        nest.Connect(self.poissonExt, self.neuronsE,
                     conn_spec=self.connDictExt,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExt})
        # Connect external input to I neurons
        nest.Connect(self.poissonExt, self.neuronsI,
                     conn_spec=self.connDictExt,
                     syn_spec={'model': 'static_synapse',
                               'weight': self.weightExt})

        # Set up connections with sparsity of self.sparsity
        # See nest documentation on 'pariwise_bernoulli' connections.
        conndict = {'rule': 'pairwise_bernoulli',
                    'p': self.sparsity}

        # EE connections
        logging.debug("Setting up EE connections.")
        nest.Connect(self.neuronsE, self.neuronsE,
                     syn_spec=self.synDictEE,
                     conn_spec=conndict)
        logging.debug("EE connections set up.")

        # EI connections
        logging.debug("Setting up EI connections.")
        nest.Connect(self.neuronsE, self.neuronsI,
                     syn_spec=self.synDictEI,
                     conn_spec=conndict)
        logging.debug("EI connections set up.")

        # II connections
        logging.debug("Setting up II connections.")
        nest.Connect(self.neuronsI, self.neuronsI,
                     syn_spec=self.synDictII,
                     conn_spec=conndict)
        logging.debug("II connections set up.")

        # IE connections
        logging.debug("Setting up IE connections.")
        nest.Connect(self.neuronsI, self.neuronsE,
                     syn_spec=self.synDictIE,
                     conn_spec=conndict)
        logging.debug("IE connections set up.")

    def setup_detector(self, population, filename, duration):
        """
        Setup spike detectors.

        :population: neuronal population to connect with
        :filename: file name of output file
        :duration: duration to detect spikes for from initialisation in s
        """
        sd_params = {
            'to_file': True,
            'label': filename,
            'stop': duration*1000.
        }

        # Create spike detectors for E and I neurons
        sd = nest.Create('spike_detector',
                         params=sd_params)
        # and connect them
        nest.Connect(population, sd)

    def prerun_setup(self):
        """Pre reun configuration."""
        self.__setup_simulation()
        self.comm.Barrier()

    def print_simulation_parameters(self):
        """Print the parameters of the simulation to a file."""
        if self.rank == 0:
            with open("00-simulation_params.txt", 'w') as pfile:
                print("{}: {} milli seconds".format("dt", self.dt),
                      file=pfile)
                print("{}: {}".format("num_E", self.populations['E']),
                      file=pfile)
                print("{}: {}".format("num_I", self.populations['I']),
                      file=pfile)
                print("{}: {}".format("num_P", self.populations['P']),
                      file=pfile)
                print("{}: {}".format("num_R", self.populations['R']),
                      file=pfile)
                print("{}: {}".format("num_colsE", self.colsE),
                      file=pfile)
                print("{}: {}".format("num_colsI", self.colsI),
                      file=pfile)
                print("{}: {} nS".format("wbar", self.wbar),
                      file=pfile)
                print("{}: {} nS".format("weightEE", self.weightEE),
                      file=pfile)
                print("{}: {} ns".format("weightPatternEE",
                                         self.weightPatternEE),
                      file=pfile)
                print("{}: {} nS".format("weightEI", self.weightEI),
                      file=pfile)
                print("{}: {} nS".format("weightII", self.weightII),
                      file=pfile)
                print("{}: {} nS".format("weightExt", self.weightExt),
                      file=pfile)
                print("{}: {}".format("sparsity", self.sparsity),
                      file=pfile)

    def __setup_simulation(self):
        """Setup the common simulation things."""
        # Nest stuff

        nest.ResetKernel()
        # http://www.nest-simulator.org/sli/setverbosity/
        nest.set_verbosity('M_INFO')
        nest.SetKernelStatus(
            {
                'resolution': self.dt,
                'overwrite_files': True
            }
        )
        self.__setup_neurons()
        self.__create_neurons()
        self.__setup_initial_connection_params()
        self.__create_network_connections()

    def __strengthen_pattern_connections(self):
        """Strengthen connections that make up the pattern."""
        connections = nest.GetConnections(source=self.pattern1,
                                          target=self.pattern1)
        nest.SetStatus(connections, {"weight": self.weightPatternEE})

        connections = nest.GetConnections(source=self.pattern2,
                                          target=self.pattern2)
        nest.SetStatus(connections, {"weight": self.weightPatternEE})

    def setup_patterns(self):
        """
        Set up the three neuron sets.

        Each is 28x28
        """
        pattern1_start = numpy.array([30, 30])
        pattern1_end = numpy.array([58, 58])
        pattern2_start = numpy.array([50, 50])
        pattern2_end = numpy.array([78, 78])
        pattern3_start = numpy.array([10, 60])
        pattern3_end = numpy.array([38, 88])
        # Lower left quarter of pattern
        recall1_start = numpy.array([30, 30])
        recall1_end = numpy.array([44, 44])

        for gid, location in self.locations.items():
            if ((location >= pattern1_start).all() and (location <=
                                                        pattern1_end).all()):
                self.pattern1.append(gid)
            if ((location >= pattern2_start).all() and (location <=
                                                        pattern2_end).all()):
                self.pattern2.append(gid)
            if ((location >= pattern3_start).all() and (location <=
                                                        pattern3_end).all()):
                self.pattern3.append(gid)
            if ((location >= recall1_start).all() and (location <=
                                                       recall1_end).all()):
                self.recall1.append(gid)

        self.__strengthen_pattern_connections()

    def setup_pattern1_for_recall(self):
        """
        Set up a pattern1 for recall.

        Creates a new poisson generator and connects it to a recall subset of
        the pattern - the poisson stimulus will run for the set recall_duration
        from the invocation of this method.
        """
        # set up external stimulus
        stim_time = nest.GetKernelStatus()['time']
        neuronDictStim = {'rate': 100.,
                          'origin': stim_time,
                          'start': 0., 'stop': self.recall_duration}
        # We only create on, but each connection in nest receives a different
        # spike train
        stim = nest.Create('poisson_generator', 1,
                           neuronDictStim)

        nest.Connect(stim, self.recall1,
                     conn_spec=self.connDictStim)


if __name__ == "__main__":
    # Set up logging configuration
    test = True
    logging.basicConfig(
        format='%(funcName)s: %(lineno)d: %(levelname)s: %(message)s',
        level=logging.DEBUG)

    # Correct NEST version
    simulation = VogelsSprekeler()

    # simulation setup
    # set up neurons, connections
    simulation.prerun_setup()
    # print em up
    simulation.print_simulation_parameters()
    logging.info("Rank {}: SIMULATION SETUP".format(simulation.rank))

    # initial setup
    logging.info("Rank {}: SIMULATION STARTED".format(simulation.rank))
    # Initial to get the neurons going
    nest.Simulate(200.)

    logging.info("Rank {}: running for Figure A".format(simulation.rank))
    # set up spike detectors to get initial spikes for Figure A
    simulation.setup_detector(simulation.neuronsE, "spikes-E-A", 5.)
    simulation.setup_detector(simulation.neuronsI, "spikes-I-A", 5.)
    nest.Simulate(1000 * 3600.)

    logging.info("Rank {}: running for Figure B".format(simulation.rank))
    # Simulate and record spikes for 5 second for Figure B
    simulation.setup_detector(simulation.neuronsE, "spikes-E-B", 5.)
    simulation.setup_detector(simulation.neuronsI, "spikes-I-B", 5.)
    nest.Simulate(1000 * 5.)

    logging.info("Rank {}: running for Figure C".format(simulation.rank))
    # Figure C
    # Mark three patterns, strengthen synapses in two of them
    simulation.setup_patterns()
    # Get spikes of all neurons for next 5 seconds
    simulation.setup_detector(simulation.neuronsE, "spikes-E-C", 5.)
    simulation.setup_detector(simulation.neuronsI, "spikes-I-C", 5.)
    nest.Simulate(1000 * 3600.)

    logging.info("Rank {}: running for Figure D".format(simulation.rank))
    # Figure D
    simulation.setup_detector(simulation.neuronsE, "spikes-E-D", 5.)
    simulation.setup_detector(simulation.neuronsI, "spikes-I-D", 5.)
    nest.Simulate(1000 * 5.)

    logging.info("Rank {}: running for Figure E".format(simulation.rank))
    # Figure E
    simulation.setup_pattern1_for_recall()
    simulation.setup_detector(simulation.neuronsE, "spikes-E-E", 5.)
    simulation.setup_detector(simulation.neuronsI, "spikes-I-E", 5.)
    nest.Simulate(1000 * 5.)

    logging.info("Rank {}: SIMULATION FINISHED SUCCESSFULLY".format(
        simulation.rank))
