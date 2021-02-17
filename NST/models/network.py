# -*- coding: utf-8 -*-
# @Date    : 2/17/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0


from .modules import AdaIN, SaAdaIN, BaseStyleNet


class AdaINNet(BaseStyleNet):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.style_norm = AdaIN()


class SaAdaINNet(BaseStyleNet):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.style_norm = SaAdaIN(512)
