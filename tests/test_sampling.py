"""Test the sampling routines from trimem.mc"""
import numpy as np
import pytest

from trimem.mc.mesh import Mesh
from trimem.mc.hmc import HMC, HMCOptions,MeshHMC, MeshFlips,MeshFlipOptions, MeshMonteCarlo

from .util import icosahedron


# ------------------------------------------------------------------------------
#                                                               reference data -
# ------------------------------------------------------------------------------
ref_hmc = np.array(
    [[1.13647863365959e+00, -1.01660312114166e-01, 4.87723745819153e-01],
    [7.43048024884800e-01, 8.77375020153709e-01, -1.63621557925122e+00],
    [-6.01296606946988e-01, 8.54870580972993e-01, -1.20642215378182e+00],
    [-2.01938274633694e-01, -4.48553058560741e-01, -3.31671062076030e-02],
    [6.00284601452301e-01, 6.26687917747375e-01, -6.34274983077240e-01],
    [2.89512545709741e-02, -1.88380610488677e-01, 9.91965524207009e-02],
    [3.86333466232453e-01, 3.30150312579298e-01, -1.47043612001433e+00],
    [-1.65218044216865e+00, 1.78947560604801e+00, -9.20891913467497e-01],
    [6.56267944949398e-02, 6.18373375839036e-01, 1.47704252681693e+00],
    [-1.00333978961422e-01, 3.08317248760165e-01, 3.53187279061704e-01],
    [3.51067019988282e-01, -5.17735259902402e-02, -1.87265234189593e+00],
    [-1.96096671486500e-02, 2.79136606603488e+00, 6.37990839051303e-01],]
)

ref_mesh_mc=np.array(
    [[-3.19581773742851e-01, 8.68223283648085e-01, -6.83332641029001e-01],
    [1.51785788749995e+00, -8.32302731150623e-02, -1.09992345703986e+00],
    [-1.18118534990196e+00, -1.56463867979178e+00, 8.33460850171300e-01],
    [-1.07853564619199e+00, -6.18801750246102e-02, -1.26720163001985e-01],
    [6.57306145172932e-01, 1.98019151354936e+00, 6.36384628668605e-01],
    [2.73152229811758e-01, 5.64030099714897e-01, 1.27222103039028e+00],
    [-5.13597811192063e-01, -1.59889673631294e+00, -1.20663913386221e+00],
    [1.51639404914838e-01, -1.83824306239546e-01, -4.59668559587055e-01],
    [2.48325824293901e-01, 4.68659279619788e-01, -3.11470449674122e-01],
    [6.76128442958748e-01, 5.07641339707812e-01, 9.91601704497391e-01],
    [-3.23157907386885e-01, 6.96110735957957e-01, 4.47174659746813e-03],
    [-3.17237604771872e-01, -1.58518585403334e-01, -5.41602543434624e-02]]
)

# ------------------------------------------------------------------------------
#                                                                        tests -
# ------------------------------------------------------------------------------
def test_hmc():
    """Test plain HMC."""
    np.random.seed(42)

    p, c = icosahedron()
    mesh = Mesh(p,c)

    # simplified energy and gradient evaluators
    def energy(x):
        return 0.5*x.ravel().dot(x.ravel())

    def gradient(x):
        return x
    
    hmc = HMC(mesh.x, energy, gradient, options=HMCOptions(info_step=10,time_step=1e-1),gen=np.random.default_rng(42))

    hmc.run(10)

    # generate new reference solution to copy-paste above
    if False:
        import io
        with io.StringIO() as fp:
            np.savetxt(fp, hmc.x, fmt=["[%.14e,", "%.14e,", "%.14e],"])
            print(fp.getvalue())

    assert np.linalg.norm(hmc.x - ref_hmc) < 1.0e-14

def test_mesh_hmc():
    """Test MeshHMC."""

    p, c = icosahedron()
    mesh = Mesh(p,c)

    # simplified energy and gradient evaluators, but with mesh access
    def energy(x):
        mesh.x = x
        return 0.5*x.ravel().dot(x.ravel())

    def gradient(x):
        mesh.x = x
        return x
    
    hmc = MeshHMC(mesh, energy, gradient, options=HMCOptions(info_step=10,time_step=1e-1),gen=np.random.default_rng(42))

    hmc.run(10)
    
    assert np.allclose(hmc.x,ref_hmc)#np.linalg.norm(hmc.x - ref_hmc) < 1.0e-14, repr((np.allclose(hmc.x,ref_hmc),hmc.x.shape,ref_hmc.shape,np.linalg.norm(hmc.x)-np.linalg.norm(ref_hmc)))

@pytest.fixture(params=["flip_parallel", "flip_serial"])
def flip_type(request):
    return request.param.split("_")[1]

def test_mesh_mc(flip_type):
    """Test full mesh monte carlo, i.e., moves + flips."""

    p, c = icosahedron()
    mesh = Mesh(p,c)

    # simplified energy and gradient evaluators, but with mesh access
    def energy(x):
        mesh.x = x
        return 0.5*x.ravel().dot(x.ravel())

    def gradient(x):
        mesh.x = x
        return x

    gen=np.random.default_rng(42)
    hmc = MeshHMC(mesh, energy, gradient, options=HMCOptions(info_step=10,time_step=1e-1),gen=gen)
    
    import trimem.core as m
    eparams = m.EnergyParams()
    estore = m.EnergyManager(mesh.trimesh, eparams)
    flips = MeshFlips(mesh,estore,options=MeshFlipOptions(flip_type=flip_type,info_step=10),gen=gen)
    mc = MeshMonteCarlo(hmc=hmc,flips=flips,gen=gen)

    mc.run(10)

    if False:
        import io
        with io.StringIO() as fp:
            np.savetxt(fp, hmc.x, fmt=["[%.14e,", "%.14e,", "%.14e],"])
            print(fp.getvalue())
    assert np.allclose(mesh.x,ref_mesh_mc)
