'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@date 11/15/18
'''

from spiders.coco_ssm_spider import COCOSSMSpiderAttentionMask8_128, COCOSSMSpiderAttentionMask8_192, COCOSSMSpiderAttentionMask16_192
from alchemy.engines.caffe_python_layers import AlchemyDataLayer

class COCOSSMSpiderAttentionMask8_128(AlchemyDataLayer):

    spider =  COCOSSMSpiderAttentionMask8_128

class COCOSSMSpiderAttentionMask8_192(AlchemyDataLayer):

    spider =  COCOSSMSpiderAttentionMask8_192
    
class COCOSSMSpiderAttentionMask16_192(AlchemyDataLayer):

    spider =  COCOSSMSpiderAttentionMask16_192    