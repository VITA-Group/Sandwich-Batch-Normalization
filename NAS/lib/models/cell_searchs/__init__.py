##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# The macro structure is defined in NAS-Bench-201
from .search_model_darts import TinyNetworkDarts
from .search_model_darts_ccbn import TinyNetworkDartsCCBN
from .search_model_darts_sabn import TinyNetworkDartsSaBN

nas201_super_nets = {
    "DARTS-V1": TinyNetworkDarts,
    "DARTS-V1-ccbn": TinyNetworkDartsCCBN,
    "DARTS-V1-sabn": TinyNetworkDartsSaBN,
}
