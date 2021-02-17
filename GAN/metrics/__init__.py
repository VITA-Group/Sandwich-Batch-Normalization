# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

from .fid_score_torch import calculate_fid_given_paths_torch
from .inception_score import get_inception_score

__all__ = ["calculate_fid_given_paths_torch", "get_inception_score"]
