__author__ = 'zgf'

import time

import nibabel as nib

from algorithm.region_growing import *
from algorithm.similarity_criteria import *
from algorithm.stop_criteria import *
from algorithm.region import *
from algorithm.optimizer import *

if __name__ == "__main__":
    #load data
    starttime = time.clock()
    image = nib.load("../data/S1/zstat1.nii.gz")
    affine = image.get_affine()
    image = image.get_data()

    seed_coords = np.array([[28, 29, 29]]) # r_OFA seed point: [28, 29, 29]

    similarity_criteria = SlicBasedSimilarityCriteria('euclidean')
    stop_criteria = StopCriteria('size')
    threshold = np.array((2, 4, 6, 8, 10))

    slic_srg = SlicSRG(similarity_criteria, stop_criteria, 10000)
    slic_image = slic_srg.get_slic_image(image)
    nib.save(nib.Nifti1Image(slic_image, affine), "../data/S1/zstat1_slic_srg_supervoxel_image.nii.gz")

    seed = slic_image[seed_coords[:, 0], seed_coords[:, 1], seed_coords[:, 2]]
    region = SlicRegion(seed, slic_image)


    result_volume = np.zeros_like(image)
    slic_srg_regions = slic_srg.compute(region, image, threshold)

    result_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(slic_srg_regions)))
    for i in range(len(slic_srg_regions)):
        labels = slic_srg_regions[i].get_label()
        result_image[labels[:, 0], labels[:, 1], labels[:, 2], i] = 1

    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_slic_srg.nii.gz")

    #AC
    optimizer_AC = Optimizer('AC')
    optimizer_AC_image = optimizer_AC.compute([slic_srg_regions], image)
    #PC
    # optimizer_PC = Optimizer('PC')
    # optimier_PC_image = optimizer_PC.compute(rsrg_region, image[..., j])

    index = optimizer_AC_image[0].argmax()
    optimal_regions = slic_srg_regions[index]
    labels = optimal_regions.get_label()

    result_image = np.zeros_like(image)
    result_image[labels[:, 0], labels[:, 1], labels[:, 2]] = 1
    nib.save(nib.Nifti1Image(result_image, affine), "../data/S1/zstat1_slic_asrg_images.nii.gz")

    endtime = time.clock()
    print 'Cost time:: ', np.round((endtime - starttime), 3), ' s'















