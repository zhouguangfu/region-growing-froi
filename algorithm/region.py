# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

'''
An object to represent the region and its associated attributes

'''

import numpy as np
import sys

class Region(object):
    '''
    inputs:
        label_cords: a 2D numpy array, represents the cordinates of the voxel in the current region.
        region_id: a list, each value in the list represents a region which been growed into the current region.
        shape: the original image shape
        voxel_value: not valid
        neighbor_region: a list, each value represents the region nerghbor to the current region

    '''

    def __init__(self, label_cords, region_id, shape, voxel_value, neighbor_region):
        self.label_cords = label_cords
        self._region_id = region_id
        self._shape = shape
        self._voxel_value = voxel_value
        self._neighbor_region = neighbor_region


    def add_region(self, region):
        self.region_values |= region.get_region_value()
        self.is_changed = True

    def remove_region(self, region):
        self.region_values ^= region.get_region_value()
        self.is_changed = True

    def get_region_value(self):
        return self.region_values

    def add_neighbor_region(self, neighbor_region):
        self.neighbor_values |= neighbor_region.get_neighbor_region_value()
        self.neighbor_values -= self.region_values

    def remove_neighbor_region(self, neighbor_region):
        self.neighbor_values -= neighbor_region.get_region_value()

    def get_neighbor_region_value(self):
        return self.neighbor_values

    def get_region_mean(self):
        if self.is_changed:
            mask = self.generate_region_mask()
            self.mean = self.image[mask].mean()
            self.is_changed = False
        return self.mean

    def get_region_values_size(self):
        return self.region_values.__len__()

    def get_region_size(self):
        mask = self.generate_region_mask()
        return mask.sum()

    def get_neighbor_values_size(self):
        return self.neighbor_values.__len__()

    def get_neighbor_size(self):
        mask = self.generate_region_neighbor_mask()
        return mask.sum()

    def generate_region_mask(self):
        region_mask = np.zeros_like(self.unique_image).astype(np.bool)
        for region_value in self.region_values:
            region_mask[self.unique_image == region_value] = True

        return region_mask

    def generate_region_neighbor_mask(self):
        neighbor_mask = np.zeros_like(self.unique_image).astype(np.bool)
        for neighbor_value in self.neighbor_values:
            neighbor_mask[self.unique_image == neighbor_value] = True

        return neighbor_mask

    def get_region_cords(self):
        dimension = len(self.image.shape)
        region_mask = self.generate_region_mask()
        region_cords = np.nonezeros(region_mask).reshape((dimension, -1))

        return region_cords

    def get_neighbor_region_cords(self):
        dimension = len(self.image.shape)
        neighbor_region_mask = self.generate_region_neighbor_mask()
        neighbor_region_cords = np.nonezeros(neighbor_region_mask).reshape((dimension, -1))

        return neighbor_region_cords

    def compute_neighbor_regions(self):
        from scipy.ndimage.morphology import binary_dilation

        region_mask = self.generate_region_mask()
        #compute new neighbors
        neighbor_mask = binary_dilation(region_mask)
        neighbor_mask[region_mask] = 0
        neighbor_values = np.unique(self.unique_image[neighbor_mask > 0])
        neighbor_values = np.delete(neighbor_values, 0)
        #Add new neighbor values
        self.neighbor_values |= set(neighbor_values.tolist())
