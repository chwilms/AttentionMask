from base_item import  BaseItem, Field


class COCOItem(BaseItem):

    imgToAnns = Field()
    image_path = Field()

    def __init__(self, image_path, imgToAnns, **kwargs):
        self.image_path = image_path
        self.imgToAnns = imgToAnns
        super(COCOItem, self).__init__(**kwargs)
