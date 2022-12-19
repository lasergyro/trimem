"""Monte Carlo Sampling.

A vanilla Hamiltonian Monte Carlo implementation with optional cooling.
Additionally, a multi-proposal Monte Carlo algorithm is available that is
capable to integrate trimem-specific edge-flip functionality as flip
proposals into the Monte Carlo framework.
"""

from typing import Literal, Optional
import numpy as np
from collections import Counter

from .. import core as m

def _vv_integration(x0, p0, force, m, dt, N):
    """Velocity verlet integration (using momentum instead of velocity)."""

    x = x0
    p = p0
    a = force(x)
    for i in range(N):
        x  = x + (p * dt + 0.5 * a * dt**2) / m
        an = force(x)
        p  = p + 0.5 * (a + an) * dt
        a  = an

    return x, p

def get_step_counters():
    """Counter to manage the accounting of steps for moves and flips.

    Returns:
        collections.Counter:
            Counter with keys for vertex-moves and edge-flips.
    """
    return Counter(move=0, flip=0)

from dataclasses import dataclass,asdict
def pprint_options(options):
    width = max([len(str(k)) for k in options.__dataclass_fields__])
    for k, v in asdict(options).items():
        print(f"  {k: <{width}}: {v}")

@dataclass(frozen=True)
class HMCOptions:
    """HMC options
    * ``mass`` (default: 1.0): scaling factor for unit-diagonal mass
              matrix used in the time integration
    * ``time_step`` (default: 1.0e-4): time step for time integration
    * ``num_integration_steps`` (default: 100): number of time
        integration steps
    * ``initial_temperature`` (default: 1.0): initial temperature for
        simulated annealing
    * ``minimal_temperature`` (default: 1.0e-6): minimal temperature
        for annealing
    * ``cooling_factor`` (default: 0.0): factor for exponential cooling
    * ``cooling_start_step`` (default: 0): start simulated annealing
        at this step
    * ``info_step`` (default: 100): print info every n'th step
    * ``init_step`` (default: 0): start value for step counter
    """
    mass : float =1.
    time_step : float = 1e-4
    num_integration_steps : int = 100
    initial_temperature : float = 1.
    minimal_temperature : float = 1e-6
    cooling_factor : float = 0.
    cooling_start_step : int = 0
    info_step : int = 100

class HMC:
    """Simple Hamiltonian Monte Carlo (with optional cooling).

    This class implements the `marching` only. Recording of the generated
    chain/trajectory must be provided by the user within the callable
    `callback` given as optional argument to the constructor. This callback
    must implement the signature `callback(x)` with `x` being an element of
    the sample space.

    Args:
        x (ndarray[float]): initial state
        nlog_prob (callable): negative log of probability density function
        grad_nlog_prob (callable): gradient of negative log of pdf

    Keyword Args:
        callback (callable): step callback with signature callback(x)
            (defaults to no-op.)
        options : HMCOptions

    """

    def __init__(
        self,
        x,
        nlog_prob,
        grad_nlog_prob,
        callback=None,
        counter=get_step_counters(),
        options=HMCOptions(),
        gen=None,
    ):
        """Initialization."""
        self.gen=gen if gen is not None else np.random.default_rng()
        # function, gradient and callback evaluation
        self.nlog_prob      = nlog_prob
        self.grad_nlog_prob = grad_nlog_prob
        self.cb             = lambda x,s: None if callback is None else callback

        # init options
        self.m     = options.mass
        self.dt    = options.time_step
        self.L     = options.num_integration_steps
        self.Tinit = options.initial_temperature
        self.Tmin  = options.minimal_temperature
        self.fT    = options.cooling_factor
        self.cN    = options.cooling_start_step
        self.istep = options.info_step

        # pretty print options
        print("\n---------------------------------------")
        print("Hamiltonian Monte Carlo Initialization:")
        pprint_options(options)
        

        # ref to step counters
        self.counter = counter

        # init algorithm
        self.i   = 0
        self.acc = 0
        self.T   = self.Tinit

        # initial state
        self.x = x

    def _hamiltonian(self,x,p):
        """Evaluate Hamiltonian."""
        return self.nlog_prob(x) + 0.5 * p.ravel().dot(p.ravel()) / self.m

    def _step(self):
        """Metropolis step."""

        # adjust momentum variance due to current temperature
        p_var = self.m*self.T

        # sample momenta
        p = self.gen.normal(size=self.x.shape)*np.sqrt(p_var)

        # integrate trajectory
        force = lambda x: -self.grad_nlog_prob(x)
        xn, pn = _vv_integration(self.x, p, force, self.m, self.dt, self.L)

        # evaluate energies
        dh = (self._hamiltonian(xn, pn) - self._hamiltonian(self.x,p)) / self.T

        # compute acceptance probability: min(1, np.exp(-de))
        a = 1.0 if dh<=0 else np.exp(-dh)
        u = self.gen.uniform()
        acc = u<=a
        if acc:
            self.x    = xn
            self.acc += 1

        # update internal step counter
        self.i += 1

    def info(self):
        """Print algorithmic information."""
        i_total = sum(self.counter.values())
        if self.istep and i_total % self.istep == 0:
            ar = self.acc/self.i if not self.i == 0 else 0.0
            print("\n-- HMC-Step ", self.counter["move"])
            print("----- acc-rate:   ", ar)
            print("----- temperature:", self.T)
            self.acc = 0
            self.i   = 0

    def step(self):
        """Make one step."""

        # update temperature
        i = sum(self.counter.values()) #py3.10: self.counter.total()
        Tn = np.exp(-self.fT * (i - self.cN)) * self.Tinit
        self.T = max(min(Tn, self.Tinit), self.Tmin)

        # make a step
        self._step()

        # update step count
        self.counter["move"] += 1

    def run(self, N):
        """Run HMC for N steps."""
        for i in range(N):
            self.step()
            self.info()
            self.cb(self.x, self.counter)


class MeshHMC(HMC):
    """HMC with mesh as state.

    A lightweight extension of :class:`HMC` that can be initialized with
    a mesh as the `state` in the sampling space.

    Args:
        x (Mesh): initial state
        nlog_prob (callable): negative log of probability density function
        grad_nlog_prob (callable): gradient of negative log of pdf

    Keyword Args:
        callback (callable): step callback with signature callback(x)
            (defaults to no-op.)
        options (dict-like): algorithm parametrization. (see :class:`HMC`)
    """

    def __init__(
        self,
        mesh, 
        nlog_prob,
        grad_nlog_prob,
        callback=None,
        counter=get_step_counters(),
        options=HMCOptions(),
        gen=None,
    ):
        """Init."""
        super().__init__(
            x=mesh.x,
            nlog_prob=nlog_prob,
            grad_nlog_prob=grad_nlog_prob,
            callback=callback,
            counter=counter,
            options=options,
            gen=gen,
        )
        self.mesh = mesh

    def step(self):
        """Make a step and explicitly update the mesh vertices."""
        super().step()
        self.mesh.x = self.x




@dataclass(frozen=True)
class MeshFlipOptions:
    """
    * ``flip_type`` (default: 'parallel'): 'serial' or 'parallel' flip evaluation
    * ``flip_ration`` (default: 0.1): proportion of edges in the mesh
        for which a flip is attempted
    * ``info_step`` (default: 100): print info every n'th step
    * ``init_step`` (default: 0): initial value for the step counter
    """
    flip_type : Literal['serial','parallel'] = 'parallel'
    flip_ratio : float = .1
    info_step : int = 100

class MeshFlips:
    """Flipping edges as a step in a Markov Chain.

    This class wraps the flip functionality available from the core
    C++-module such that it fits into a multi-proposal Monte Carlo framework.

    Args:
        mesh (Mesh): initial state.
        estore (EnergyManager): `backend` for performing flipping on edges.

    Keyword Args:
        options : MeshFlipOptions

            
    """
    def __init__(
        self,
        mesh,
        estore,
        counter=get_step_counters(),
        options=MeshFlipOptions(),
        gen=None,
    ):
        """Init."""

        self.mesh   = mesh
        self.estore = estore

        # init options
        self.istep = options.info_step
        self.fr    = options.flip_ratio
        self.ft    = options.flip_type
        self.gen=gen if gen is not None else np.random.default_rng()


        # pretty print options
        print("\n--------------------------------")
        print("Flips Monte Carlo Initialization:")
        pprint_options(options)

        if self.ft == "none" or self.fr == 0.0:
            self._flips = lambda: 0
        elif self.ft == "serial":
            self._flips = lambda: m.flip(
                self.mesh.trimesh, self.estore, self.fr,self.gen.integers(0,high=np.iinfo(np.int32).max))
        elif self.ft == "parallel":
            self._flips = lambda: m.pflip(
                self.mesh.trimesh, self.estore, self.fr,self.gen.integers(0,high=np.iinfo(np.int32).max))
        else:
            raise ValueError("Wrong flip-type: {}".format(self.ft))

        self.i   = 0
        self.acc = 0
        self.counter = counter

    def info(self):
        """Print algorithmic information."""
        i_total = sum(self.counter.values())
        if self.istep and i_total % self.istep == 0:
            n_edges = self.mesh.trimesh.n_edges()
            ar      = self.acc / (self.i * n_edges) if not self.i == 0 else 0.0
            print("\n-- MCFlips-Step ", self.counter["flip"])
            print("----- flip-accept: ", ar)
            print("----- flip-rate:   ", self.fr)
            self.acc = 0
            self.i   = 0

    def step(self):
        """Make one step."""
        self.acc += self._flips()
        self.i += 1
        self.counter["flip"] += 1

    def run(self, N):
        """Make N flip-sweeps."""
        for i in range(N):
            self.step()
            self.info()


class MeshMonteCarlo:
    """MonteCarlo with two-step moves.

    Bundles :class:`HMC` and :class:`MeshFlips` into a bi-step Monte Carlo
    algorithm where each step comprises a step of the :class:`HMC` or
    a step of :class:`MeshFlips` with equal probability.

    Args:
        hmc (HMC): HMC algorithm
        flips (MeshFlips): Monte Carlo flip algorithm

    Keyword Args:
        callback (callable): step callback with signature callback(x,s) with
            x begin the state and s being compatible with collections.Counter
            (defaults to no-op.)
    """

    def __init__(
        self,
        hmc,
        flips,
        counter=get_step_counters(),
        callback=None,
        gen=None,
    ):
        """Initialize."""
        self.gen=gen if gen is not None else np.random.default_rng()

        self.hmc   = hmc
        self.flips = flips
        self.cb    = (lambda x, s: None) if callback is None else callback

        # make counters consistent (! works only for mutables)
        self.counter       = counter
        self.hmc.counter   = counter
        self.flips.counter = counter

    def step(self):
        """Make one step each with each algorithm."""
        if self.gen.choice(2) == 0:
            self.hmc.step()
        else:
            self.flips.step()

    def run(self, N):
        """Run for N steps."""
        for i in range(N):
            self.step()
            self.hmc.info()
            self.flips.info()
            self.cb(self.flips.mesh.x, self.counter)
