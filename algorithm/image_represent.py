# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

'''
An class to convert the original image to the represent regions

'''

import numpy as np
import segment
from region import Region

class ImageRepresent(object):
    '''
    inputs:
        image: the original image to be represented.
        methods: 'slic' or 'watershed'
        mask: only convert the areas to regions which the mask represented. the left is zero.

        sigma: default is 1.0, needed when the methods is 'watershed'
        threshed: default is 2.3, needed when the methods is 'watershed'\

        n_segmentation: default is 10000, needed when the methods is 'slic',
                        represented the number of the supervoxels based on the original image.

    outputs:
        regions: the regions which can represented the original image.

    '''

    def __init__(self, image, methods, mask, sigma=1.0, threshed=2.3, n_segmentation=None):
        self._image = image
        self._methods = methods
        self._mask = mask

        #optinonal parameters
        self._sigma = 1.0
        self._threshed = 2.3
        self._n_segmentation = n_segmentation

        #inner member
        #which is the bridge of the original image and the output regions
        self._unique_image = None
        #the output result
        self._regions = None


    # def __init__(self, label_cords, region_id, shape, voxel_value, neighbor_region):
    #     self.label_cords = label_cords
    #     self._region_id = region_id
    #     self._shape = shape
    #     self._voxel_value = voxel_value
    #     self._neighbor_region = neighbor_region

    def convert_image_to_regions(self, image):
        '''
        Convert the original image to regions

        '''

        if self._methods == 'watershed':
            self._unique_image = self.get_watershed_image(image)
        elif self._methods == 'slic':
            self._unique_image = self.get_slic_image(image)
        else:
            self._unique_image = None
            raise ValueError('Please input the correct methods parameters.')

        values_len = (self._unique_image > 0).sum()
        regions = [None * (values_len - 1)]
        unique_values = np.unique(self._unique_image)

        for region_id in range(1, unique_values.size):
            # print 'i: ', i, '   generate region...'
            regions[region_id - 1] = Region(None,
                                            region_id,
                                            self._image.shape,
                                            None,
                                            None)
        print 'Regions: ', regions

        return regions

    def get_region_from_index(self, index):
        '''
        Get corresponding region from certain index.

        '''
        if index < 0 or index > len(self._regions):
            raise ValueError('get_region_from_index: index is invalid!')

        if self._regions != None:
            return self._regions[index]
        else:
            return None

    #-------------------- Complete the get_region_from_seeds function!!!! -----------------------
    def get_region_from_seeds(self, seed_cords):
        '''
        Get corresponding region from seeds.
        Inputs:
              seed_cords: the cordinates of the seed

        '''
        if not isinstance(seed_cords, np.ndarray):
            raise ValueError('get_region_from_seeds: get_region_from_seeds is not np.ndarray!')

        region = None
        if self._unique_image != None:
            region_ids = self._unique_image[seed_cords[:, 0], seed_cords[:, 1], seed_cords[:, 2]]
            if len(region_ids) == 1:
                region = Region(seed_cords, region_ids[0], self._image.shape, None, None)
            else:
                for region_id in region_ids:
                    sub_region = self.get_region_from_index(region_id - 1)
                    region.add_region()

        return region


    def get_watershed_image(self, lines):
        '''
        Convert the image to unique_image(parcel) using watershed method.

        '''
        sigma = 1.0
        threshed = 5.0
        sfx = segment.inverse_transformation

        new_image = self._image.copy()
        if self._mask != None:
            new_image[self._mask == False] = 0

        watershed_image = segment.watershed(new_image, sigma, threshed, None, sfx)
        if self._mask != None:
            watershed_image[self._mask == False] = 0

        return watershed_image


    def get_slic_image(self, lines):
        '''
        Convert the image to unique_image(supervoxel) using slic method.

        '''
        from skimage.segmentation import slic

        new_image = self._image.copy()
        if self._mask != None:
            new_image[self._mask == False] = 0

        #Convert the original to the 0~255 gray image
        gray_image = (new_image - new_image.min()) * 255 / (new_image.max() - new_image.min())
        slic_image = slic(gray_image.astype(np.float),
                          n_segments=self._n_segmentation,
                          slic_zero=True,
                          sigma=2,
                          multichannel=False,
                          enforce_connectivity=True)
        return slic_image






