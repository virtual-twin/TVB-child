from tvb.basic.neotraits.api import HasTraits, NArray, Attr, Range
from tvb.simulator.history import SparseHistory
from tvb.simulator.common import simple_gen_astr
from tvb.simulator.coupling import SparseCoupling

import numpy as np

class EIB_Linear(SparseCoupling):
    r"""
    Provides a modified linear coupling function for EI Tuning, scaling the pre values by w which can be a scalar or a matrix. 

    .. math::
        a (w*x) + b

    """

    a = NArray(
        label=":math:`a`",
        default=np.array([0.00390625,]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="Rescales the connection strength while maintaining the ratio "
            "between different values.")

    b = NArray(
        label=":math:`b`",
        default=np.array([0.0]),
        doc="Shifts the base of the connection strength while maintaining "
            "the absolute difference between different values.")
    
    w = NArray(
        label=":math:`a`",
        default=np.array([1.,]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="Balancing of long-range excitatory and feedforward inhibitory synaptic. Needs to be defined and have the shape (nodes, 2, nodes)")

    parameter_names = 'a b w'.split()
    pre_expr = 'x_j * w'
    post_expr = 'a * gx + b'

    def post(self, gx):
        return self.a * gx + self.b

    def __str__(self):
        return simple_gen_astr(self, 'a b')

from tvb.basic.neotraits.api import NArray, Final, List, Range
from tvb.simulator.models.base import ModelNumbaDfun

class EIB_ReducedWongWangExcInh(ModelNumbaDfun):
    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2014] Deco Gustavo, Ponce Alvarez Adrian, Patric Hagmann,
                  Gian Luca Romani, Dante Mantini, and Maurizio Corbetta. *How Local
                  Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics*.
                  The Journal of Neuroscience 34(23), 7886 –7898, 2014.


    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}) \\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))} \\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, {\gamma}H(x_{ek}) \\

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + {\lambda}GJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}) \\
                 H(x_{ik})    &=  \dfrac{a_ix_{ik} - b_i}{1 - \exp(-d_i(a_ix_{ik} -b_i))} \\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \

    """

    # Define traited attributes for this model, these represent possible kwargs.

    a_e = NArray(
        label=":math:`a_e`",
        default=np.array([310., ]),
        domain=Range(lo=0., hi=500., step=1.),
        doc="[n/C]. Excitatory population input gain parameter, chosen to fit numerical solutions.")

    b_e = NArray(
        label=":math:`b_e`",
        default=np.array([125., ]),
        domain=Range(lo=0., hi=200., step=1.),
        doc="[Hz]. Excitatory population input shift parameter chosen to fit numerical solutions.")

    d_e = NArray(
        label=":math:`d_e`",
        default=np.array([0.160, ]),
        domain=Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Excitatory population input scaling parameter chosen to fit numerical solutions.""")

    gamma_e = NArray(
        label=r":math:`\gamma_e`",
        default=np.array([0.641/1000, ]),
        domain=Range(lo=0.0, hi=1.0/1000, step=0.01/1000),
        doc="""Excitatory population kinetic parameter""")

    tau_e = NArray(
        label=r":math:`\tau_e`",
        default=np.array([100., ]),
        domain=Range(lo=50., hi=150., step=1.),
        doc="""[ms]. Excitatory population NMDA decay time constant.""")

    w_p = NArray(
        label=r":math:`w_p`",
        default=np.array([1.4, ]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population recurrence weight""")

    J_N = NArray(
        label=r":math:`J_N`",
        default=np.array([0.15, ]),
        domain=Range(lo=0.001, hi=0.5, step=0.001),
        doc="""[nA] NMDA current""")

    W_e = NArray(
        label=r":math:`W_e`",
        default=np.array([1.0, ]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population external input scaling weight""")

    a_i = NArray(
        label=":math:`a_i`",
        default=np.array([615., ]),
        domain=Range(lo=0., hi=1000., step=1.),
        doc="[n/C]. Inhibitory population input gain parameter, chosen to fit numerical solutions.")

    b_i = NArray(
        label=":math:`b_i`",
        default=np.array([177.0, ]),
        domain=Range(lo=0.0, hi=200.0, step=1.0),
        doc="[Hz]. Inhibitory population input shift parameter chosen to fit numerical solutions.")

    d_i = NArray(
        label=":math:`d_i`",
        default=np.array([0.087, ]),
        domain=Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Inhibitory population input scaling parameter chosen to fit numerical solutions.""")

    gamma_i = NArray(
        label=r":math:`\gamma_i`",
        default=np.array([1.0/1000, ]),
        domain=Range(lo=0.0, hi=2.0/1000, step=0.01/1000),
        doc="""Inhibitory population kinetic parameter""")

    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=np.array([10., ]),
        domain=Range(lo=5., hi=100., step=1.0),
        doc="""[ms]. Inhibitory population NMDA decay time constant.""")

    J_i = NArray(
        label=r":math:`J_{i}`",
        default=np.array([1.0, ]),
        domain=Range(lo=0.001, hi=2.0, step=0.001),
        doc="""[nA] Local inhibitory current""")

    W_i = NArray(
        label=r":math:`W_i`",
        default=np.array([0.7, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory population external input scaling weight""")

    I_o = NArray(
        label=":math:`I_{o}`",
        default=np.array([0.382, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external input""")

    I_ext = NArray(
        label=":math:`I_{ext}`",
        default=np.array([0.0, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external stimulus input""")

    G = NArray(
        label=":math:`G`",
        default=np.array([2.0, ]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling""")

    lamda = NArray(
        label=r":math:`\lambda`",
        default=np.array([0.0, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory global coupling scaling""")

    state_variable_range = Final(
        default={
            "S_e": np.array([0.0, 1.0]),
            "S_i": np.array([0.0, 1.0])
        },
        label="State variable ranges [lo, hi]",
        doc="Population firing rate")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"S_e": np.array([0.0, 1.0]), "S_i": np.array([0.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""")

    coupling_terms = Final(
        label="Coupling terms",
        # how to unpack coupling array
        default=["c_0", "c_1"]
    )

    state_variable_dfuns = Final(
        label="Drift functions",
        default={
            "coupling": "G * J_N * c_0", # LRE
            "coupling1": "G * J_N * c_1",# FFI
            "J_N_S_e": "J_N * S_e",
            "_x_e": "w_p * J_N_S_e - J_i * S_i + W_e * I_o + coupling + I_ext",
            "x_e": "a_e * _x_e - b_e",
            "H_e": "x_e / (1 - exp(-d_e * x_e))",
            "S_e": "- (S_e / tau_e) + (1 - S_e) * H_e * gamma_e",
            "_x_i": "J_N_S_e - S_i + W_i * I_o + lamda * coupling1",
            "x_i": "a_i * _x_i - b_i",
            "H_i": "x_i / (1 - exp(-d_i * x_i))",
            "S_i": "- (S_i / tau_i) + H_i * gamma_i"
        }
    )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('S_e', 'S_i'),
        default=('S_e', 'S_i'),
        doc="""default state variables to be monitored""")
    
    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default="a_e b_e d_e gamma_e tau_e w_p J_N W_e a_i b_i d_i gamma_i tau_i J_i W_i I_o I_ext G lamda".split())

    state_variables = ['S_e', 'S_i']
    _nvar = 2
    cvar = np.array([0], dtype=np.int32)

    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        pass