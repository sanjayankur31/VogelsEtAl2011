#!/usr/bin/env python3
"""
Post process spike files.

File: postprocess.py

Copyright 2018 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>

The pipeline is broken into steps so that one can inspect the outputs of each
step if needed.
"""

import numpy
import collections
import pandas


def calculate_firing_rate(neuronset, spikes_fn, neuron_locations, snap_time):
    """Get firing rates for individual neurons at a particular point in time.

    The output file format is:
    nid xcor ycor rate

    :neuronset: name of neuron set (E/I)
    :spikes_fn: name of input spikes file
    :neuron_locations: three column array: nid xcor ycor
    :snap_time: time for which snapshot will be generated (in seconds)
    :returns: True if everything went OK, else False

    """
    # Window to count spikes over
    window = 5000.
    start = 0.
    end = 0.

    # read spike files
    chunk = pandas.read_csv(spikes_fn, sep='\s+',
                            names=["neuronID", "spike_time"],
                            dtype={'neuronID': numpy.uint16,
                                   'spike_time': float},
                            lineterminator="\n",
                            skipinitialspace=True,
                            header=None, index_col=None)

    spikes = numpy.array(chunk.values[:, 0])
    times = numpy.array(chunk.values[:, 1])

    snap_time *= 1000.

    # Find our values
    start = numpy.searchsorted(times,
                               snap_time - window,
                               side='left')
    end = numpy.searchsorted(times,
                             snap_time,
                             side='right')
    # Required time not found in this file at all
    if start == len(times):
        return False
    else:
        neurons = spikes[start:end]
        spike_counts = collections.Counter(neurons)

        # Fill up neurons that may not have spiked
        rates = {}
        for neuron in neuron_locations:
            n_id = int(neuron[0])
            if n_id not in spike_counts:
                rates[n_id] = 0
            else:
                rates[n_id] = spike_counts[n_id]/(window/1000)

        o_fn = "firing-rates-{}-{}.gdf".format(
            neuronset, snap_time/1000.)

        with open(o_fn, 'w') as fh:
            for neuron in neuron_locations:
                print("{}\t{}\t{}\t{}".format(
                    neuron[0], neuron[1], neuron[2],
                    rates[int(neuron[0])]), file=fh)

    return True


if __name__ == "__main__":
    neurons_E = (numpy.loadtxt("00-neuron-locations-E.txt", delimiter='\t',
                               usecols=[0, 1, 2], skiprows=1))
    neurons_I = (numpy.loadtxt("00-neuron-locations-I.txt", delimiter='\t',
                               usecols=[0, 1, 2], skiprows=1))

    # Get values for figure A
    calculate_firing_rate('E', 'spikes-E-A.gdf', neurons_E, 5.)
    calculate_firing_rate('I', 'spikes-I-A.gdf', neurons_I, 5.)

    # Get values for figure B
    calculate_firing_rate('E', 'spikes-E-B.gdf', neurons_E, 3610.)
    calculate_firing_rate('I', 'spikes-I-B.gdf', neurons_I, 3610.)

    # Get values for figure C: patterns stored
    calculate_firing_rate('E', 'spikes-E-C.gdf', neurons_E, 3615.)
    calculate_firing_rate('I', 'spikes-I-C.gdf', neurons_I, 3615.)

    # Get values for figure D
    calculate_firing_rate('E', 'spikes-E-D.gdf', neurons_E, 7220.)
    calculate_firing_rate('I', 'spikes-I-D.gdf', neurons_I, 7220.)

    # Get values for figure E
    calculate_firing_rate('E', 'spikes-E-E.gdf', neurons_E, 7225.)
    calculate_firing_rate('I', 'spikes-I-E.gdf', neurons_I, 7225.)
