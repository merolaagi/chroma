"""
CHROMA
Chromatin-Regulated Hierarchy of Reference-frame Object Modules
with Action-conditioned prediction.

Module map, keyed to the specification:

    groups.py       Sec 1   exact unitary rep of Z^2 x| C_n  (Prop 0)
    predictor.py    Sec 2   rho(d)[s + Delta_phi]            (Prop 1, Prop 4)
    memory.py       Sec 3   canonicalisation                 (Prop 2)
    regulatory.py   Sec 4   attractors, enhancers, D_m       (Prop 3)
    losses.py       Sec 5   L_pred, L_loop, L_vote, L_reg + collapse metrics
    voting.py       Sec 6   pushforward voting, disagreement
    train.py        Sec 7   four-phase developmental curriculum
    experiments.py  Sec 8   E1-E4
    world.py        Sec 9   minimal tactile instance
"""

from .groups import GroupAction, SE2Rep
from .encoder import EMATarget, PatchEncoder
from .predictor import EquivariantResidual, Predictor
from .memory import CanonicalMemory
from .regulatory import (Differentiation, EnhancerReadout, FlatRegulator,
                         MLPRegulator, RegulatoryState)
from .voting import HypothesisGrid, VotingBus
from .model import CHROMA, ChromaConfig, PHASES
from .train import Trainer, TrainConfig
from .world import TactileWorld
from .search import (ActionSet, InfoGainPolicy, MCTSPolicy, RandomPolicy)

__version__ = "0.1.0"
__all__ = [
    "SE2Rep", "GroupAction", "PatchEncoder", "EMATarget", "Predictor",
    "EquivariantResidual", "CanonicalMemory", "RegulatoryState",
    "MLPRegulator", "FlatRegulator", "EnhancerReadout", "Differentiation", "HypothesisGrid",
    "VotingBus", "CHROMA", "ChromaConfig", "PHASES", "Trainer", "TrainConfig",
    "TactileWorld", "ActionSet", "InfoGainPolicy", "MCTSPolicy",
    "RandomPolicy",
]
