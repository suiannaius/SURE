def ZScoreNorm(image, seg=None, use_mask_for_norm=None, num_modalities=4):
    
    assert image.shape[0] == num_modalities, f"Expected first dimension to be {num_modalities} (modalities), but got {image.shape[0]}."
    if use_mask_for_norm:
        assert seg is not None, f"When using mask for normalization, the ground truth should be provided."
        assert image.shape == seg.shape, f"The shapes of image and label should be the same, but got {image.shape} and {seg.shape}."
        mask = seg >= 0
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / (max(std, 1e-8))
    else:
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (max(std, 1e-8))
    
    return image
