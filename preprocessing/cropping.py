import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert data.ndim in (3, 4), "Data must have shape (C, X, Y, Z) or (C, X, Y), i.e., channel first."
    
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask) # (240, 240, 155)
    
    return nonzero_mask


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    nonzero_mask = create_nonzero_mask(data) # (240, 240, 155)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    data = data[tuple([slice(None), *slicer])]
    if seg is not None:
        seg = seg[tuple([slice(None), *slicer])]
    nonzero_mask = nonzero_mask[slicer][None]
    if seg is not None:
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label # No data at this pixel -> background.
    else:
        nonzero_mask = nonzero_mask.astype(np.int16)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
        
    return data, seg, bbox
