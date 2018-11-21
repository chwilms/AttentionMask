from base_dataset import BaseDataset, Dummy
from alchemy.items.coco import COCOItem
from pycocotools import coco

import config


class COCO_DS(BaseDataset, coco.COCO):

    def __init__(self, annotation_file, ign_null_img=False):
        BaseDataset.__init__(self)
        self.annotation_file = annotation_file
        coco.COCO.__init__(self, annotation_file)

        for i in self.getImgIds():
            image_file_name = config.IMAGE_NAME_FORMAT % (config.IMAGE_SET, i)
            image_file_path = config.IMAGE_PATH_FORMAT % (config.IMAGE_SET, image_file_name)
            try:
                if len(self.imgToAnns[i]) == 0:
                    raise KeyError
                self.append(COCOItem(image_file_path, self.imgToAnns[i]))
            except KeyError:
                if ign_null_img is False:
                    self.append(COCOItem(image_file_path, []))
