import sys
import os
import time

sys.path.append("/Users/.../ABAGAIL/jython/ABAGAIL.jar")

#import java.io.FileReader as FileReader
#import java.io.File as File
#import java.lang.String as String
#import java.lang.StringBuffer as StringBuffer
#import java.lang.Boolean as Boolean
#import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array

from time import clock
from itertools import product
from base import *
from pathlib import Path

"""
Commandline parameter(s):
   none
"""

N=100  # Number of local optima
T=N/10
fill = [2] * N
ranges = array('i', fill)

maxIters = 2500
numTrials = 5

# module_path = os.path.dirname(os.path.realpath(__file__))
# output_path = os.path.join(module_path, 'output_file')
# outfile = module_path + '/CONTPEAKS/CONTPEAKS_{}_{}_LOG.csv'
# outfile = output_path + '\CONTPEAKS\CONTPEAKS_{}_{}_LOG.csv'
# outfile = 'C:\\Users\\gablanco\\gatech\\assignment2\\CONTPEAKS\\CONTPEAKS_{}_{}_LOG.csv'

outfile = 'CONTPEAKS_{}_{}_LOG.csv'

ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

#rhc = RandomizedHillClimbing(hcp)
#fit = FixedIterationTrainer(rhc, 200000)
#fit.train()
#print "RHC: " + str(ef.value(rhc.getOptimal()))

# RHC

for t in range(numTrials):
    fname = outfile.format('RHC', str(t + 1))
    with open(fname, 'w') as f:
        f.write('iterations,fitness,time,fevals\n')
    ef = ContinuousPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 100)
    times = [0]
    for i in range(0, maxIters, 10):
        start = clock()
        fit.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        fevals = ef.getFunctionEvaluations()
        score = ef.value(rhc.getOptimal())
        #ef.fevals -= 1
        fevals -= 1
        st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
        print st
        with open(fname, 'a') as f:
            f.write(st)

# SA
for t in range(numTrials):
    for CE in [0.15, 0.35, 0.55, 0.75, 0.95]:
        fname = outfile.format('SA{}'.format(CE), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        ef = ContinuousPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(1E10, CE, hcp)
        fit = FixedIterationTrainer(sa, 10)
        times = [0]
        for i in range(0, maxIters, 10):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            fevals = ef.getFunctionEvaluations()
            score = ef.value(sa.getOptimal())
            fevals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print st
            with open(fname, 'a') as f:
                f.write(st)

# GA
for t in range(numTrials):
    for pop, mate, mutate in product([100], [50, 30, 10], [50, 30, 10]):
        fname = outfile.format('GA{}_{}_{}'.format(pop, mate, mutate), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        ef = ContinuousPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
        fit = FixedIterationTrainer(ga, 10)
        times = [0]
        for i in range(0, maxIters, 10):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            fevals = ef.getFunctionEvaluations()
            score = ef.value(ga.getOptimal())
            fevals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print st
            with open(fname, 'a') as f:
                f.write(st)

# MIMIC
for t in range(numTrials):
    for samples, keep, m in product([100], [50], [0.1, 0.3, 0.5, 0.7, 0.9]):
        fname = outfile.format('MIMIC{}_{}_{}'.format(samples, keep, m), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        ef = ContinuousPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        df = DiscreteDependencyTree(m, ranges)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        mimic = MIMIC(samples, keep, pop)
        fit = FixedIterationTrainer(mimic, 10)
        times = [0]
        for i in range(0, maxIters, 10):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            fevals = ef.getFunctionEvaluations()
            score = ef.value(mimic.getOptimal())
            fevals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print st
            with open(fname, 'a') as f:
                f.write(st)


sa = SimulatedAnnealing(1E11, .95, hcp)
fit = FixedIterationTrainer(sa, 200000)
fit.train()
print "SA: " + str(ef.value(sa.getOptimal()))

ga = StandardGeneticAlgorithm(200, 100, 10, gap)
fit = FixedIterationTrainer(ga, 1000)
fit.train()
print "GA: " + str(ef.value(ga.getOptimal()))

mimic = MIMIC(200, 20, pop)
fit = FixedIterationTrainer(mimic, 1000)
fit.train()
print "MIMIC: " + str(ef.value(mimic.getOptimal()))
