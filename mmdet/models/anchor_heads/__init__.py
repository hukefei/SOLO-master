from .anchor_head import AnchorHead
from .atss_head import ATSSHead
from .fcos_head import FCOSHead
from .fovea_head import FoveaHead
from .free_anchor_retina_head import FreeAnchorRetinaHead
from .ga_retina_head import GARetinaHead
from .ga_rpn_head import GARPNHead
from .guided_anchor_head import FeatureAdaption, GuidedAnchorHead
from .reppoints_head import RepPointsHead
from .retina_head import RetinaHead
from .retina_sepbn_head import RetinaSepBNHead
from .rpn_head import RPNHead
from .ssd_head import SSDHead
from .solo_head import SOLOHead
from .decoupled_solo_head import DecoupledSOLOHead
from .decoupled_solo_light_head import DecoupledSOLOLightHead
from .solov2_head import SOLOV2Head
from .solov2_head_add import SOLOV2HeadADD
# from .solo_attention_head import SOLOAttentionHead
from .solov2_head_test import SOLOAttentionHead

__all__ = [
    'AnchorHead', 'GuidedAnchorHead', 'FeatureAdaption', 'RPNHead',
    'GARPNHead', 'RetinaHead', 'RetinaSepBNHead', 'GARetinaHead', 'SSDHead',
    'FCOSHead', 'RepPointsHead', 'FoveaHead', 'FreeAnchorRetinaHead',
    'ATSSHead', 'SOLOHead', 'DecoupledSOLOHead', 'DecoupledSOLOLightHead',
    'SOLOV2Head', 'SOLOAttentionHead', 'SOLOV2HeadADD'
]
