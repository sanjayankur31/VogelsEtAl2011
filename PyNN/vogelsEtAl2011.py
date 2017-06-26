
"""
Model implementation in PyNN by Vitor Chaud, Andrew Davison and Padraig Gleeson (July 2013).
This implementation extends the original implementation to reproduce Fig. 4 of the paper describing the model

Original implementation reference:

	Inhibitory synaptic plasticity in a
	recurrent network model (F. Zenke, 2011)

Adapted from:

	Vogels, T. P., H. Sprekeler, F. Zenke, C. Clopath, and W. Gerstner. 'Inhibitory Plasticity Balances
	Excitation and Inhibition in Sensory Pathways and Memory Networks.' Science (November 10, 2011).
"""

#############################################
##
##	VERSION 0.2 - Using PyNN 0.8
##
#############################################


from pyNN.random import RandomDistribution, NumpyRNG
from pyNN.utility import get_script_args, Timer, ProgressBar, init_logging, normalized_filename
import matplotlib.pyplot as plt
from auxRoutines import *
from pyNN.nest import native_synapse_type

simulator_name = get_script_args(1)[0]

exec("from pyNN.{} import *".format(simulator_name))

print("\n")
print("Starting PyNN with simulator: {}".format(simulator_name))

timer = Timer()

# Total of 8000 excitatory neurons and 2000 inhibitory neurons. Note that all these specified number of neurons in a population are disjoint.
# For instance pattern1 population has a total of 720 neurons, 180 of which comprises pattern1_stim.

numOfNeuronsExcPopulation = 5712 	# Excitatory neurons not including pattern1, pattern2 and patternIntersection populations
numOfNeuronsPattern1 = 524		# Neurons of pattern1 not including those of patternIntersection and from neuronsPattern1_stim
numOfNeuronsPattern1_stim = 196		# Sub-set of pattern1 which may receive external stimulation
numOfNeuronsPattern2 = 524		# Neurons of pattern2 not including those of patternIntersection
numOfNeuronsPattern2_stim = 196		# Sub-set of pattern2 which may receive external stimulation
numOfNeuronsPatternIntersection = 64
numOfNeuronsControl = 784

numOfNeuronsInhibPopulation = 2000


numOfSampledNeuronsPattern1_ISICV = 392
numOfSampledNeuronsControl_ISICV = 392
numOfSampledNeuronsPattern1_corr = 15
numOfSampledNeuronsControl_corr = 15





connectivity = 0.02
weightExcSynapses = 0.003 		# [uS]
weightInhibToInhibSynapses = 0.03 	# [uS]
potentiationFactor = 5

timeStep = 0.1


setup(timestep=timeStep, min_delay=0.5)



tau_m 		= 20.0	# [ms]
cm 		= 0.2 	# [nF]
v_rest 		= -60.0	# [mV]
v_thresh 	= -50.0 # [mV]
tau_syn_E 	= 5.0	# [ms]
tau_syn_I 	= 10.0	# [ms]
e_rev_E 	= 0.0	# [mV]
e_rev_I 	= -80.0	# [mV]
v_reset 	= -60.0	# [mV]
tau_refrac 	= 5.0	# [ms]
i_offset 	= 0.2	# [nA]



eta = 1e-4
eta = weightInhibToInhibSynapses * eta  # weight of inhibitory to excitatory synapses
rho = 0.003

synapseDelay = 0.5 # [ms]


neuronParameters = 	{
			'tau_m':	tau_m,
			'cm':		cm,
			'v_rest':	v_rest,
			'v_thresh':	v_thresh,
			'tau_syn_E':	tau_syn_E,
			'tau_syn_I':	tau_syn_I,
			'e_rev_E':	e_rev_E,
			'e_rev_I':	e_rev_I,
			'v_reset':	v_reset,
			'tau_refrac':	tau_refrac,
			'i_offset': 	i_offset
			}


cell_type = IF_cond_exp(**neuronParameters)



# Simulation time setup

timePreSim 	= 60000 		# 1 min (60000 ms)
timeSimFig4A 	= 1000000
timeSimFig4B 	= 2600000		# 60 min (60 * 60 * 1000 ms)
timeSimFig4C 	= 5000			# 5 sec (5000 ms)
timeSimFig4D 	= 3600000 - 5000 	# 60 min (60 * 60 * 1000 ms)
timeSimFig4E_part1 = 1000
timeSimFig4E_part2 = 4000		# 5 sec (5000 ms)


timeBoundKernel = 300


### SIMULATION TIMES WERE DOWNSCALED FOR TESTING PURPOSES

downscaleFactor = 100
minSimTime = 50 #[ms]

eta = eta * downscaleFactor

timePreSim 	= int(round(timePreSim/downscaleFactor))
timeSimFig4A 	= int(round(timeSimFig4A/downscaleFactor))
timeSimFig4B 	= int(round(timeSimFig4B/downscaleFactor))
timeSimFig4C 	= int(round(timeSimFig4C/downscaleFactor))
timeSimFig4D 	= int(round(timeSimFig4D/downscaleFactor))
timeSimFig4E_part1 = int(round(timeSimFig4E_part1/downscaleFactor))
timeSimFig4E_part2 = int(round(timeSimFig4E_part2/downscaleFactor))

if timePreSim <= minSimTime:
	timePreSim = minSimTime
if timeSimFig4A <= minSimTime:
	timeSimFig4A = minSimTime
if timeSimFig4B <= minSimTime:
	timeSimFig4B = minSimTime
if timeSimFig4C <= minSimTime:
	timeSimFig4C = minSimTime
if timeSimFig4D <= minSimTime:
	timeSimFig4D = minSimTime
if timeSimFig4E_part1 <= minSimTime:
	timeSimFig4E_part1 = minSimTime
if timeSimFig4E_part2 <= minSimTime:
	timeSimFig4E_part2 = minSimTime







timer.start()

print("\n")
print("-------------------------------------------------")
print("---------- Creating neuron populations ----------")
print("-------------------------------------------------")

excPopulation 		= Population(numOfNeuronsExcPopulation		, cell_type, label='excPop')

inhibPopulation 	= Population(numOfNeuronsInhibPopulation	, cell_type, label='inhibPop')

pattern1 		= Population(numOfNeuronsPattern1		, cell_type, label='pattern1')

pattern1_stim 		= Population(numOfNeuronsPattern1_stim		, cell_type, label='pattern1_stim')

pattern2 		= Population(numOfNeuronsPattern2		, cell_type, label='pattern2')

pattern2_stim 		= Population(numOfNeuronsPattern2_stim		, cell_type, label='pattern2_stim')

patternIntersection 	= Population(numOfNeuronsPatternIntersection	, cell_type, label='patternIntersection')

controlPopulation	= Population(numOfNeuronsControl		, cell_type, label='controlPop')


stimulus = Population(1000, SpikeSourcePoisson(rate=100.0))


print("-> DONE")
print("----------------------------------------------------------------------")
print("---------- Initialising membrane potential to random values ----------")
print("----------------------------------------------------------------------")

rand_distr = RandomDistribution('uniform', (v_reset, v_thresh), rng=NumpyRNG(seed=85524))

excPopulation.initialize(v=rand_distr)
inhibPopulation.initialize(v=rand_distr)
pattern1.initialize(v=rand_distr)
pattern1_stim.initialize(v=rand_distr)
pattern2.initialize(v=rand_distr)
pattern2_stim.initialize(v=rand_distr)
patternIntersection.initialize(v=rand_distr)
controlPopulation.initialize(v=rand_distr)

print("-> DONE")

# allow self-connections??

# what are the initial conditions??

# what are the synaptic delays?

progress_bar = ProgressBar(width=30)

fpc 	= FixedProbabilityConnector(connectivity, callback=progress_bar)

exc_synapse_type 	= native_synapse_type('static_synapse')(**{'weight': weightExcSynapses, 'delay': synapseDelay})
inhibitory_stdp_synapse_type 	= native_synapse_type('vogels_sprekeler_synapse')(**{'weight': -0.00001, 'Wmax': -3000., 'alpha': .12, 'eta': 0.01, 'tau': 20.})

print("-----------------------------------------------")
print("------- Creating excitatory projections -------")
print("-----------------------------------------------")

connections={}

print('\ne_to_e: ')
connections['e_to_e'] 		= Projection(excPopulation, 	excPopulation, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\ne_to_p1: ')
connections['e_to_p1'] 		= Projection(excPopulation, 	pattern1, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\ne_to_p1_stim: ')
connections['e_to_p1_stim'] 	= Projection(excPopulation, 	pattern1_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\ne_to_p2: ')
connections['e_to_p2']		= Projection(excPopulation, 	pattern2, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\ne_to_p2_stim: ')
connections['e_to_p2_stim'] 	= Projection(excPopulation, 	pattern2_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\ne_to_pi: ')
connections['e_to_pi'] 		= Projection(excPopulation, 	patternIntersection, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\ne_to_c: ')
connections['e_to_c'] 		= Projection(excPopulation, 	controlPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\ne_to_i: ')
connections['e_to_i'] 		= Projection(excPopulation, 	inhibPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')


print('\np1_to_e: ')
connections['p1_to_e'] 		= Projection(pattern1, 		excPopulation, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_to_p1: ')
connections['p1_to_p1']		= Projection(pattern1, 		pattern1, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_to_p1_stim: ')
connections['p1_to_p1_stim'] 	= Projection(pattern1, 		pattern1_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_to_p2: ')
connections['p1_to_p2']		= Projection(pattern1, 		pattern2, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_to_p2_stim: ')
connections['p1_to_p2_stim'] 	= Projection(pattern1, 		pattern2_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_to_pi: ')
connections['p1_to_pi']		= Projection(pattern1, 		patternIntersection, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_to_c: ')
connections['p1_to_c'] 		= Projection(pattern1, 		controlPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_to_i: ')
connections['p1_to_i'] 		= Projection(pattern1, 		inhibPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')


print('\np1_stim_to_e: ')
connections['p1_stim_to_e'] 		= Projection(pattern1_stim, 		excPopulation, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_stim_to_p1: ')
connections['p1_stim_to_p1']		= Projection(pattern1_stim, 		pattern1, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_stim_to_p1_stim: ')
connections['p1_stim_to_p1_stim'] 	= Projection(pattern1_stim, 		pattern1_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_stim_to_p2: ')
connections['p1_stim_to_p2']		= Projection(pattern1_stim, 		pattern2, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_stim_to_p2_stim: ')
connections['p1_stim_to_p2_stim'] 	= Projection(pattern1_stim, 		pattern2_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_stim_to_pi: ')
connections['p1_stim_to_pi']		= Projection(pattern1_stim, 		patternIntersection, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_stim_to_c: ')
connections['p1_stim_to_c'] 		= Projection(pattern1_stim, 		controlPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np1_stim_to_i: ')
connections['p1_stim_to_i'] 		= Projection(pattern1_stim, 		inhibPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')


print('\np2_to_e: ')
connections['p2_to_e'] 		= Projection(pattern2, 		excPopulation, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_to_p1: ')
connections['p2_to_p1']		= Projection(pattern2, 		pattern1, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_to_p1_stim: ')
connections['p2_to_p1_stim'] 	= Projection(pattern2, 		pattern1_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_to_p2: ')
connections['p2_to_p2']		= Projection(pattern2, 		pattern2, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_to_p2_stim: ')
connections['p2_to_p2_stim'] 	= Projection(pattern2, 		pattern2_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_to_pi: ')
connections['p2_to_pi']		= Projection(pattern2, 		patternIntersection, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_to_c: ')
connections['p2_to_c'] 		= Projection(pattern2, 		controlPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_to_i: ')
connections['p2_to_i'] 		= Projection(pattern2, 		inhibPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')


print('\np2_stim_to_e: ')
connections['p2_stim_to_e'] 		= Projection(pattern2_stim, 		excPopulation, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_stim_to_p1: ')
connections['p2_stim_to_p1']		= Projection(pattern2_stim, 		pattern1, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_stim_to_p1_stim: ')
connections['p2_stim_to_p1_stim'] 	= Projection(pattern2_stim, 		pattern1_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_stim_to_p2: ')
connections['p2_stim_to_p2']		= Projection(pattern2_stim, 		pattern2, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_stim_to_p2_stim: ')
connections['p2_stim_to_p2_stim'] 	= Projection(pattern2_stim, 		pattern2_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_stim_to_pi: ')
connections['p2_stim_to_pi']		= Projection(pattern2_stim, 		patternIntersection, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_stim_to_c: ')
connections['p2_stim_to_c'] 		= Projection(pattern2_stim, 		controlPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\np2_stim_to_i: ')
connections['p2_stim_to_i'] 		= Projection(pattern2_stim, 		inhibPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')


print('\npi_to_e: ')
connections['pi_to_e'] 		= Projection(patternIntersection, 		excPopulation, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\npi_to_p1: ')
connections['pi_to_p1']		= Projection(patternIntersection, 		pattern1, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\npi_to_p1_stim: ')
connections['pi_to_p1_stim'] 	= Projection(patternIntersection, 		pattern1_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\npi_to_p2: ')
connections['pi_to_p2']		= Projection(patternIntersection, 		pattern2, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\npi_to_p2_stim: ')
connections['pi_to_p2_stim'] 	= Projection(patternIntersection, 		pattern2_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\npi_to_pi: ')
connections['pi_to_pi']		= Projection(patternIntersection, 		patternIntersection, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\npi_to_c: ')
connections['pi_to_c'] 		= Projection(patternIntersection, 		controlPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\npi_to_i: ')
connections['pi_to_i'] 		= Projection(patternIntersection, 		inhibPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')


print('\nc_to_e: ')
connections['c_to_e'] 		= Projection(controlPopulation, excPopulation, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\nc_to_p1: ')
connections['c_to_p1'] 		= Projection(controlPopulation, pattern1, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\nc_to_p1_stim: ')
connections['c_to_p1_stim'] 	= Projection(controlPopulation, pattern1_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\nc_to_p2: ')
connections['c_to_p2'] 		= Projection(controlPopulation, pattern2, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\nc_to_p2_stim: ')
connections['c_to_p2_stim'] 	= Projection(controlPopulation, pattern2_stim, 		fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\nc_to_pi: ')
connections['c_to_pi'] 		= Projection(controlPopulation, patternIntersection, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\nc_to_c: ')
connections['c_to_c'] 		= Projection(controlPopulation, controlPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')
print('\nc_to_i: ')
connections['c_to_i'] 		= Projection(controlPopulation, inhibPopulation, 	fpc, 	exc_synapse_type, receptor_type='excitatory')

print("\n")
print("-----------------------------------------------")
print("------- Creating inhibitory projections -------")
print("-----------------------------------------------")

print('i_to_e: ')
connections['i_to_e'] 		= Projection(inhibPopulation, 	excPopulation, 		fpc, 	inhibitory_stdp_synapse_type, receptor_type='inhibitory')
print('\ni_to_p1: ')
connections['i_to_p1'] 		= Projection(inhibPopulation, 	pattern1, 		fpc, 	inhibitory_stdp_synapse_type, receptor_type='inhibitory')
print('\ni_to_p1_stim: ')
connections['i_to_p1_stim'] 	= Projection(inhibPopulation, 	pattern1_stim, 		fpc, 	inhibitory_stdp_synapse_type, receptor_type='inhibitory')
print('\ni_to_p2: ')
connections['i_to_p2'] 		= Projection(inhibPopulation, 	pattern2, 		fpc, 	inhibitory_stdp_synapse_type, receptor_type='inhibitory')
print('\ni_to_p2_stim: ')
connections['i_to_p2_stim'] 	= Projection(inhibPopulation, 	pattern2_stim, 		fpc, 	inhibitory_stdp_synapse_type, receptor_type='inhibitory')
print('\ni_to_pi: ')
connections['i_to_pi'] 		= Projection(inhibPopulation, 	patternIntersection, 	fpc, 	inhibitory_stdp_synapse_type, receptor_type='inhibitory')
print('\ni_to_c: ')
connections['i_to_c'] 		= Projection(inhibPopulation, 	controlPopulation, 	fpc, 	inhibitory_stdp_synapse_type, receptor_type='inhibitory')
print('\ni_to_i: ')
connections['i_to_i'] 		= Projection(inhibPopulation, 	inhibPopulation, 	fpc, 	inhibitory_stdp_synapse_type, receptor_type='inhibitory') # "eta" should be always zero










## Start
## 	The asynchronous irregular network dynamics of the model published in
## 	Vogels and Abbott (2005) without inhibitory plasticity.
## 	Original simulation time: 1 min (60000 ms)


excPopulation.record('spikes')
pattern1.record('spikes')
pattern1_stim.record('spikes')
pattern2.record('spikes')
pattern2_stim.record('spikes')
patternIntersection.record('spikes')
controlPopulation.record('spikes')
inhibPopulation.record('spikes')


buildCPUTime = timer.diff()

print("\n\nTime to build the network: %s seconds" %buildCPUTime)

print("\n--- Pre-simulation ---")
print("\nPre-simulation time: %s milliseconds" %timePreSim)


run(timePreSim)


simCPUTime_pre = timer.diff()

print("\nTime to perform the pre-simulation: %d seconds (%0.2f minutes)" %(simCPUTime_pre, simCPUTime_pre/60))

excSpikes			= 	excPopulation.get_data(		'spikes', clear="true")
pattern1Spikes 			=	pattern1.get_data(		'spikes')
pattern1_stimSpikes 		=	pattern1_stim.get_data(		'spikes', clear="true")
pattern2Spikes 			=	pattern2.get_data(		'spikes', clear="true")
pattern2_stimSpikes 		=	pattern2_stim.get_data(		'spikes', clear="true")
patternIntersectionSpikes 	=	patternIntersection.get_data(	'spikes', clear="true")
controlSpikes 			=	controlPopulation.get_data(	'spikes')
inhibSpikes 			= 	inhibPopulation.get_data(	'spikes', clear="true")


sampledPopPattern1_corr = pattern1.sample(numOfSampledNeuronsPattern1_corr)
sampledPopControl_corr = controlPopulation.sample(numOfSampledNeuronsControl_corr)

sampledPopPattern1_ISICV = pattern1.sample(numOfSampledNeuronsPattern1_ISICV)
sampledPopControl_ISICV = controlPopulation.sample(numOfSampledNeuronsControl_ISICV)

sampledPopPattern1Spikes_corr = sampledPopPattern1_corr.get_data(	'spikes', clear="true")
sampledPopControlSpikes_corr = sampledPopControl_corr.get_data(		'spikes', clear="true")

#sampledPopPattern1Spikes_ISICV = sampledPopPattern1_ISICV.get_data(	'spikes', clear="true")
#sampledPopControlSpikes_ISICV = sampledPopControl_ISICV.get_data(	'spikes', clear="true")


plt.ion()
fig = plt.figure(2, facecolor='white')

simTimeIni = 0
simTimeFin = timePreSim


im = plotFig4Column(fig, 1, timeStep, simTimeIni, simTimeFin, timeBoundKernel,
		excSpikes, pattern1Spikes, pattern1_stimSpikes, pattern2Spikes, pattern2_stimSpikes, patternIntersectionSpikes, controlSpikes, inhibSpikes,
		numOfSampledNeuronsPattern1_corr, sampledPopPattern1Spikes_corr, numOfSampledNeuronsControl_corr, sampledPopControlSpikes_corr)


#fig.subplots_adjust(left=0.8)
cbar_ax = fig.add_axes([0.08, 0.73, 0.01, 0.15])
#cbar_ax.set_ylabel('Rate [Hz]')

cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 50, 100, 150, 200])
cbar.ax.set_yticklabels(['0', '50', '100', '150', '200'])
cbar.ax.set_ylabel('Rate [Hz]')
cbar.ax.yaxis.labelpad = -50

plt.show(block=False)
fig.canvas.draw()

## Fig. 4, A
##
## 	Inhibitory to excitatory synapses are turned to 0 efficacy
##	The network is forced out of the AI regime and begins to fire at high rates
## 	Inhibitory plasticity is turned on.
## 	Original simulation time:

print("\n")
print("--------------------------------------")
print("Starting simulation to generate Fig. 4")
print("--------------------------------------")


connections['i_to_e'].set(weight=0)
connections['i_to_p1'].set(weight=0)
connections['i_to_p1_stim'].set(weight=0)
connections['i_to_p2'].set(weight=0)
connections['i_to_p2_stim'].set(weight=0)
connections['i_to_pi'].set(weight=0)
connections['i_to_c'].set(weight=0)


connections['i_to_e'].set(eta=eta)
connections['i_to_p1'].set(eta=eta)
connections['i_to_p1_stim'].set(eta=eta)
connections['i_to_p2'].set(eta=eta)
connections['i_to_p2_stim'].set(eta=eta)
connections['i_to_pi'].set(eta=eta)
connections['i_to_c'].set(eta=eta)



print("\nSimulation time: %s milliseconds" %timeSimFig4A)

run(timeSimFig4A)


simCPUTime_4A = timer.diff()

print("\nTime to perform the simulation: %d seconds (%0.2f minutes)" %(simCPUTime_4A, simCPUTime_4A/60))


excSpikes			= 	excPopulation.get_data(		'spikes', clear="true")
pattern1Spikes 			=	pattern1.get_data(		'spikes')
pattern1_stimSpikes 		=	pattern1_stim.get_data(		'spikes', clear="true")
pattern2Spikes 			=	pattern2.get_data(		'spikes', clear="true")
pattern2_stimSpikes 		=	pattern2_stim.get_data(		'spikes', clear="true")
patternIntersectionSpikes 	=	patternIntersection.get_data(	'spikes', clear="true")
controlSpikes 			=	controlPopulation.get_data(	'spikes')
inhibSpikes 			= 	inhibPopulation.get_data(	'spikes', clear="true")


sampledPopPattern1Spikes_corr = sampledPopPattern1_corr.get_data(	'spikes', clear="true")
sampledPopControlSpikes_corr = sampledPopControl_corr.get_data(		'spikes', clear="true")



print("\nPloting Fig. 4A...")


simTimeIni = timePreSim
simTimeFin = timePreSim + timeSimFig4A


plotFig4Column(fig, 2, timeStep, simTimeIni, simTimeFin, timeBoundKernel,
		excSpikes, pattern1Spikes, pattern1_stimSpikes, pattern2Spikes, pattern2_stimSpikes, patternIntersectionSpikes, controlSpikes, inhibSpikes,
		numOfSampledNeuronsPattern1_corr, sampledPopPattern1Spikes_corr, numOfSampledNeuronsControl_corr, sampledPopControlSpikes_corr)

#plt.show(block=False)
fig.canvas.draw()

## Fig. 4, B
##
## 	Inhibitory plasticity has restored asynchronous irregular dynamics
##	Original simulation time: 60 min (60 * 60 * 1000 ms)


print("\nContinuing simulation...")


print("\nSimulation time: %s milliseconds" %timeSimFig4B)

run(timeSimFig4B)





simCPUTime_4B = timer.diff()

print("\nTime to perform the simulation: %d seconds (%0.2f minutes)" %(simCPUTime_4B, simCPUTime_4B/60))

excSpikes			= 	excPopulation.get_data(		'spikes', clear="true")
pattern1Spikes 			=	pattern1.get_data(		'spikes')
pattern1_stimSpikes 		=	pattern1_stim.get_data(		'spikes', clear="true")
pattern2Spikes 			=	pattern2.get_data(		'spikes', clear="true")
pattern2_stimSpikes 		=	pattern2_stim.get_data(		'spikes', clear="true")
patternIntersectionSpikes 	=	patternIntersection.get_data(	'spikes', clear="true")
controlSpikes 			=	controlPopulation.get_data(	'spikes')
inhibSpikes 			= 	inhibPopulation.get_data(	'spikes', clear="true")


sampledPopPattern1Spikes_corr = sampledPopPattern1_corr.get_data(	'spikes', clear="true")
sampledPopControlSpikes_corr = sampledPopControl_corr.get_data(		'spikes', clear="true")


print("\nPloting Fig. 4B...")

simTimeIni = timePreSim + timeSimFig4A
simTimeFin = timePreSim + timeSimFig4A +timeSimFig4B


plotFig4Column(fig, 3, timeStep, simTimeIni, simTimeFin, timeBoundKernel,
		excSpikes, pattern1Spikes, pattern1_stimSpikes, pattern2Spikes, pattern2_stimSpikes, patternIntersectionSpikes, controlSpikes, inhibSpikes,
		numOfSampledNeuronsPattern1_corr, sampledPopPattern1Spikes_corr, numOfSampledNeuronsControl_corr, sampledPopControlSpikes_corr)

#plt.show(block=False)
fig.canvas.draw()

## Fig. 4, C
##
## 	The excitatory non-zero weights of the two designated memory patterns
## 	are increased ad-hoc by a factor of 5. The neurons of hte subset begin
## 	to exhibit elevated and more sychronized activity
##	Original simulation time: 5 sec (5000 ms)



connections['p1_to_p1'].set(weight = weightExcSynapses * potentiationFactor)
connections['p1_to_p1_stim'].set(weight = weightExcSynapses * potentiationFactor)
connections['p1_to_pi'].set(weight = weightExcSynapses * potentiationFactor)


connections['p1_stim_to_p1'].set(weight = weightExcSynapses * potentiationFactor)
connections['p1_stim_to_p1_stim'].set(weight = weightExcSynapses * potentiationFactor)
connections['p1_stim_to_pi'].set(weight = weightExcSynapses * potentiationFactor)


connections['p2_to_p2'].set(weight = weightExcSynapses * potentiationFactor)
connections['p2_to_p2_stim'].set(weight = weightExcSynapses * potentiationFactor)
connections['p2_to_pi'].set(weight = weightExcSynapses * potentiationFactor)


connections['p2_stim_to_p2'].set(weight = weightExcSynapses * potentiationFactor)
connections['p2_stim_to_p2_stim'].set(weight = weightExcSynapses * potentiationFactor)
connections['p2_stim_to_pi'].set(weight = weightExcSynapses * potentiationFactor)


connections['pi_to_p1'].set(weight = weightExcSynapses * potentiationFactor)
connections['pi_to_p1_stim'].set(weight = weightExcSynapses * potentiationFactor)
connections['pi_to_p2'].set(weight = weightExcSynapses * potentiationFactor)
connections['pi_to_p2_stim'].set(weight = weightExcSynapses * potentiationFactor)
connections['pi_to_pi'].set(weight = weightExcSynapses * potentiationFactor)


print("\nContinuing simulation...")

print("\nSimulation time: %s milliseconds" %timeSimFig4C)

run(timeSimFig4C)



simCPUTime_4C = timer.diff()

print("\nTime to perform the simulation: %d seconds (%0.2f minutes)" %(simCPUTime_4C, simCPUTime_4C/60))


excSpikes			= 	excPopulation.get_data(		'spikes', clear="true")
pattern1Spikes 			=	pattern1.get_data(		'spikes')
pattern1_stimSpikes 		=	pattern1_stim.get_data(		'spikes', clear="true")
pattern2Spikes 			=	pattern2.get_data(		'spikes', clear="true")
pattern2_stimSpikes 		=	pattern2_stim.get_data(		'spikes', clear="true")
patternIntersectionSpikes 	=	patternIntersection.get_data(	'spikes', clear="true")
controlSpikes 			=	controlPopulation.get_data(	'spikes')
inhibSpikes 			= 	inhibPopulation.get_data(	'spikes', clear="true")

sampledPopPattern1Spikes_corr = sampledPopPattern1_corr.get_data(	'spikes', clear="true")
sampledPopControlSpikes_corr = sampledPopControl_corr.get_data(		'spikes', clear="true")


print("\nPloting Fig. 4C...")

simTimeIni = timePreSim + timeSimFig4A +timeSimFig4B
simTimeFin = timePreSim + timeSimFig4A +timeSimFig4B+ timeSimFig4C


plotFig4Column(fig, 4, timeStep, simTimeIni, simTimeFin, timeBoundKernel,
		excSpikes, pattern1Spikes, pattern1_stimSpikes, pattern2Spikes, pattern2_stimSpikes, patternIntersectionSpikes, controlSpikes, inhibSpikes,
		numOfSampledNeuronsPattern1_corr, sampledPopPattern1Spikes_corr, numOfSampledNeuronsControl_corr, sampledPopControlSpikes_corr)

fig.canvas.draw()

## Fig. 4, D
##
## 	Inhibitory plasticity has succesfully suppressed any elevated activity
## 	from the pattern and restored the global background state
##	Original simulation time: 60 min (60 * 60 * 1000 ms)


print("\nContinuing simulation...")

print("\nSimulation time: %s milliseconds" %timeSimFig4D)

run(timeSimFig4D)


simCPUTime_4D = timer.diff()

print("\nTime to perform the simulation: %d seconds (%0.2f minutes)" %(simCPUTime_4D, simCPUTime_4D/60))

print("\nPloting Fig. 4D...")


excSpikes			= 	excPopulation.get_data(		'spikes', clear="true")
pattern1Spikes 			=	pattern1.get_data(		'spikes')
pattern1_stimSpikes 		=	pattern1_stim.get_data(		'spikes', clear="true")
pattern2Spikes 			=	pattern2.get_data(		'spikes', clear="true")
pattern2_stimSpikes 		=	pattern2_stim.get_data(		'spikes', clear="true")
patternIntersectionSpikes 	=	patternIntersection.get_data(	'spikes', clear="true")
controlSpikes 			=	controlPopulation.get_data(	'spikes')
inhibSpikes 			= 	inhibPopulation.get_data(	'spikes', clear="true")

sampledPopPattern1Spikes_corr = sampledPopPattern1_corr.get_data(	'spikes', clear="true")
sampledPopControlSpikes_corr = sampledPopControl_corr.get_data(		'spikes', clear="true")

simTimeIni = timePreSim + timeSimFig4A +timeSimFig4B + timeSimFig4C
simTimeFin = timePreSim + timeSimFig4A +timeSimFig4B + timeSimFig4C + timeSimFig4D


plotFig4Column(fig, 5, timeStep, simTimeIni, simTimeFin, timeBoundKernel,
		excSpikes, pattern1Spikes, pattern1_stimSpikes, pattern2Spikes, pattern2_stimSpikes, patternIntersectionSpikes, controlSpikes, inhibSpikes,
		numOfSampledNeuronsPattern1_corr, sampledPopPattern1Spikes_corr, numOfSampledNeuronsControl_corr, sampledPopControlSpikes_corr)

fig.canvas.draw()

## Fig. 4, E
##
## 	By delivering an additional, 1 s long stimulus as described above to
## 	25% of the cells within one memory pattern, the whole pattern is activated.
## 	Activity inside the pattern stays asynchronous and irregular, and the rest
## 	of the network, including the other pattern, ramains nearly unaffected
##	Original simulation time: 5 sec (5000 ms)







fpcStim 	= FixedProbabilityConnector(0.05, callback=progress_bar)

connections['stim_to_pattern1_stim'] 	= Projection(stimulus, 	pattern1_stim, 		fpcStim, 	exc_synapse_type, receptor_type='excitatory')



print("\nContinuing simulation...")

print("\nSimulation time: %s milliseconds" %timeSimFig4E_part1)

run(timeSimFig4E_part1)


simCPUTime_4E_part1 = timer.diff()

print("\nTime to perform the simulation: %d seconds (%0.2f minutes)" %(simCPUTime_4E_part1, simCPUTime_4E_part1/60))

'''
connections['stim_to_subPopPattern1Stim'].set(weight = 0)
'''

connections['stim_to_pattern1_stim'].set(weight = 0)


print("\nContinuing simulation...")

print("\nSimulation time: %s milliseconds" %timeSimFig4E_part2)

run(timeSimFig4E_part2)


simCPUTime_4E_part2 = timer.diff()

print("\nTime to perform the simulation: %d seconds (%0.2f minutes)" %(simCPUTime_4E_part2, simCPUTime_4E_part2/60))

print("\nploting Fig. 4E")


excSpikes			= 	excPopulation.get_data(		'spikes', clear="true")
pattern1Spikes 			=	pattern1.get_data(		'spikes')
pattern1_stimSpikes 		=	pattern1_stim.get_data(		'spikes', clear="true")
pattern2Spikes 			=	pattern2.get_data(		'spikes', clear="true")
pattern2_stimSpikes 		=	pattern2_stim.get_data(		'spikes', clear="true")
patternIntersectionSpikes 	=	patternIntersection.get_data(	'spikes', clear="true")
controlSpikes 			=	controlPopulation.get_data(	'spikes')
inhibSpikes 			= 	inhibPopulation.get_data(	'spikes', clear="true")


sampledPopPattern1Spikes_corr = sampledPopPattern1_corr.get_data(	'spikes', clear="true")
sampledPopControlSpikes_corr = sampledPopControl_corr.get_data(		'spikes', clear="true")


simTimeIni = timePreSim + timeSimFig4A +timeSimFig4B + timeSimFig4C + timeSimFig4D
simTimeFin = timePreSim + timeSimFig4A +timeSimFig4B + timeSimFig4C + timeSimFig4D + timeSimFig4E_part1 + timeSimFig4E_part2

im = plotFig4Column(fig, 6, timeStep, simTimeIni, simTimeFin, timeBoundKernel,
		excSpikes, pattern1Spikes, pattern1_stimSpikes, pattern2Spikes, pattern2_stimSpikes, patternIntersectionSpikes, controlSpikes, inhibSpikes,
		numOfSampledNeuronsPattern1_corr, sampledPopPattern1Spikes_corr, numOfSampledNeuronsControl_corr, sampledPopControlSpikes_corr)

fig.canvas.draw()
plt.show()


print("\nTotal simulated time: %s milliseconds" %simTimeFin)

totalCPUTime = simCPUTime_pre + simCPUTime_4A + simCPUTime_4B + simCPUTime_4C + simCPUTime_4D + simCPUTime_4E_part1 + simCPUTime_4E_part2

print("\nTotal CPU time: %d seconds (%0.2f minutes)" %(totalCPUTime, totalCPUTime/60))

raw_input("Simulation finished... Press enter to exit...")



