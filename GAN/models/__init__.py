# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

from .autogan_32 import Generator as AutoganGen32
from .sngan_32 import Discriminator as SnganDis32, Generator as SnganGen32
from .sngan_small_128 import Discriminator as SnganDis128, Generator as SnganGen128

__all__ = ["AutoganGen32", "SnganGen32", "SnganDis32", "SnganGen128", "SnganDis128"]
