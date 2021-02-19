##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from typing import List, Text

__all__ = ['change_key', 'get_cell_based_tiny_net', 'get_search_spaces']

# useful modules
from config_utils import dict2config
from .SharedUtils import change_key


# Cell-based NAS Models
def get_cell_based_tiny_net(config):
    if isinstance(config, dict): config = dict2config(config, None)  # to support the argument being a dict
    super_type = getattr(config, 'super_type', 'basic')
    group_names = ['DARTS-V1', 'DARTS-V1-ccbn', 'DARTS-V1-sabn']
    if super_type == 'basic' and config.name in group_names:
        from .cell_searchs import nas201_super_nets as nas_super_nets
        return nas_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space,
                                           config.affine, config.track_running_stats)
    else:
        raise ValueError('invalid network name : {:}'.format(config.name))


# obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
def get_search_spaces(xtype, name) -> List[Text]:
    if xtype == 'cell':
        from .cell_operations import SearchSpaceNames
        assert name in SearchSpaceNames, 'invalid name [{:}] in {:}'.format(name, SearchSpaceNames.keys())
        return SearchSpaceNames[name]
    else:
        raise ValueError('invalid search-space type is {:}'.format(xtype))
