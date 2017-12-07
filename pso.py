#!/usr/bin/python
# coding: UTF-8
#
# Author: Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
# Feel free to contact for any information.
#
# Last update: 04/02/2017
#
# You can cite this code by referencing:
#   D. Laszuk, "Python implementation of Particle
#   Swarm Optimisation," 2015-,
#   [Online] Available: http://www.laszukdawid.com/codes
#
# LICENCE:
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#

from __future__ import print_function

import logging
import numpy as np

from scipy.integrate import ode

def halton_sequence(size, dim):
    seq = np.zeros( (dim,size) )
    primeGen = next_prime()
    next(primeGen)

    for d in range(dim):
        base = next(primeGen)
        seq[d] = np.array([vdc(i, base) for i in xrange(size)])

    return seq

def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if (num % i) == 0: return False
        return True

    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2

def vdc(n, base=2):
    vdc, denom = 0,1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / float(denom)
    return vdc

class Particle:
    def __init__(self, dim=10):
        self.__dim = dim

class PSO:

    logger = logging.getLogger(__name__)

    def __init__(self, func, bounds, initPos=None, nPart=None):
        """
        Performs Particle Swarm Optimisation to find the minimum of
        `func` function within provided parameters `bounds` range.

        The `func` should be a callable, and `bounds` a list-like
        where the first and second indices contain min and max bounds,
        respectively.

        Additionally one can set up `initPos` initial positions (see
        self.setInitPos) and the number `nPart` of used particles.
        """

        # Params
        self.epsError = 1.
        self.maxGen = 6000
        self.minRepetition = 100
        self.maxRepetition = 100

        self.w = 0.01
        self.phiP = 0.2  # Local best
        self.phiG = 0.1 # Global best
        self.phiN = 0.01 # Noise affect

        # Function to be minimised
        self.problem = func

        # Set up boundary values
        self.minBound = np.array(bounds[0])
        self.maxBound = np.array(bounds[1])

        self.dim = len(bounds[0])

        # Noise preparation
        self.nMean = np.zeros(self.dim)
        cov = np.sqrt(self.maxBound - self.minBound)/2
        self.nCov = np.diag(cov)
        MN = np.random.multivariate_normal
        cov[cov==0] = 1
        norm = np.sum(cov)*2*np.pi
        self.noise = lambda: (MN(self.nMean, self.nCov, 1)/norm).flatten()

        # Speed vector
        self.speed = np.ones(self.dim)
        # Number of particles in swarm
        if nPart is None:
            nPart = 100
        self.nPart = nPart

        # Initial positions
        if initPos is not None:
            initPos = np.array(initPos).reshape((-1,self.dim))
        self.initPos = initPos

        # How often print debug update
        self.refreshRate = 1 # %

    def setInitPos(self, initPos):
        """
        Variable `initPos` sets initial values for the PSO.
        It is a 2D list of (iParts, params) shape.

        If `iParts` is smaller than self.nPart then it will initiate
        first iParts with provided parameters and the rest particles
        will be assigned with random params within boundaries.
        """
        initPos = np.array(initPos).reshape((-1,self.dim))
        self.initPos = initPos

    def __initPart(self):
        """Initiate particles"""

        _size = initPos.shape[0] if self.initPos is not None else 0
        self.seq = halton_sequence(self.nPart-_size+3, self.dim)

        # Create particles
        self.Particles = [Particle(self.dim) for i in range(self.nPart)]

        # Position in generated semi-random sequence
        seqPos = 0

        # Initiate pos and fit for particles
        for part in self.Particles:

            # Initial position
            if self.initPos == None:

                part.pos = self.seq[:,seqPos]*(self.maxBound-self.minBound)
                part.pos += self.minBound
                seqPos += 1
            else:
                part.pos = self.initPos[0,:]
                self.initPos = np.delete(self.initPos, 0,0)

                # If nothing left on initial pos
                if len(self.initPos) == 0: self.initPos = None

            # Initial velocity
            part.vel = np.random.random(self.dim)*(np.sqrt(self.maxBound - self.minBound)/2)
            part.vel *= [-1., 1.][np.random.random()>0.5]

            # Initial fitness
            part.fitness = self.problem(part.pos)
            part.bestFit = part.fitness
            part.bestPos = part.pos

        # Global best fitness
        self.globBestFit = self.Particles[0].fitness
        self.globBestPos = self.Particles[0].pos
        for part in self.Particles:
            if part.fitness < self.globBestFit:
                self.globBestFit = part.fitness
                self.globBestPos = part.pos

    def update(self):
        """Updates each step"""
        for part in self.Particles:

            # Gen param
            rP, rG = np.random.random(2)
            # Replacing random values with speed vector

            w, phiP, phiG = self.w, self.phiP, self.phiG

            # Update velocity
            v, pos = part.vel, part.pos
            part.vel = self.w*v
            part.vel += phiP*rP*(part.bestPos-pos) # local best update
            part.vel += phiG*rG*(self.globBestPos-pos) # global best update

            part.vel += self.phiN*self.noise() # perturbation

            # New position
            part.pos += part.vel

            # If pos outside bounds
            if np.any(part.pos<self.minBound):
                idx = part.pos<self.minBound
                part.pos[idx] = self.minBound[idx]
            if np.any(part.pos>self.maxBound):
                idx = part.pos>self.maxBound
                part.pos[idx] = self.maxBound[idx]

            # New fitness
            part.fitness = self.problem(part.pos)

        # Global and local best fitness
        for part in self.Particles:

            # Comparing to local best
            if part.fitness < part.bestFit:
                part.bestFit = part.fitness

            # Comparing to global best
            if part.fitness < self.globBestFit:
                self.globBestFit = part.fitness
                self.globBestPos = part.pos

    def getGlobalBest(self):
        return self.globBestPos, self.globBestFit

    def optimize(self):
        """ Optimisation function.
            Before it is run, initial values should be set.
        """

        # Initiate particles
        self.__initPart()
        self.lastGlobBestFit = 0
        self.changeCounter = 0

        idx = 0
        while(idx < self.maxGen):
            if idx % int(self.maxGen*self.refreshRate*0.01) == 0:
                self.logger.debug("Gen: {}/{}  -- best = {}".format(idx, self.maxGen, self.globBestFit))

            # Perform search
            self.update()

            # Acceptably close to solution
            if self.globBestFit < self.epsError:
                return self.getGlobalBest()

            if self.globBestFit == self.lastGlobBestFit and \
                idx > self.minRepetition:

                self.changeCounter += 1
                if self.changeCounter == self.maxRepetition:
                    self.logger.debug("Obtained limit of repeating the same value.")
                    self.logger.debug("Stopping calculations.")
                    break
            else:
                self.changeCounter = 0
            self.lastGlobBestFit = self.globBestFit

            # next gen
            idx += 1

        # Search finished
        return self.getGlobalBest()


#################################

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    N = 1000
    t = np.linspace(-2, 2, N)

    S = 4*t*np.cos( 4*np.pi*t**2) * np.exp(-3*t**2)
    S+= (np.random.random(N)-0.5)*0.05

    rec = lambda a: a[0]*t*np.sin(a[1]*np.pi*t**2 +a[2])*np.exp(-a[3]*t**2)
    minProb = lambda a: np.sum(np.abs(S-rec(a))**2)

    numParam = 4
    bounds = ([0]*numParam, [10]*numParam)

    pso = PSO(minProb, bounds)
    bestPos, bestFit = pso.optimize()

    print('bestFit: ', bestFit)
    print('bestPos: ', bestPos)

    ############################
    # Visual results representation
    import pylab as plt
    plt.figure()
    plt.plot(t, S, 'b')
    plt.plot(t, rec(bestPos), 'r')
    plt.xlabel("Time")
    plt.ylabel("Amp")
    plt.title("Input (blue) and reconstuction (red)")

    plt.savefig('fit',dpi=120)
    plt.show()

