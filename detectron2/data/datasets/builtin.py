# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from .refreasoning import register_counter_refreasoning


def register_all_refreasoning_sg_counter(dataroot, debug=False):
    anno_folder, img_folder = 'expression', 'images'
    for sample_method in ['foil_rel_len16', 'foil_rel_len11']:
        for split in ['train']:
            setname = 'refreason_prop_counter_{}_{}'.format(sample_method, split)
            if debug: setname += '_debug'
            register_counter_refreasoning(setname, dataroot, anno_folder, img_folder, split,
                                          debug=debug, sample_method=sample_method)

    sample_method = 'foil_rel'
    for split in ['val','test']:
        setname = 'refreason_prop_counter_{}_{}'.format(sample_method, split)
        if debug: setname += '_debug'
        register_counter_refreasoning(setname, dataroot, anno_folder, img_folder, split,
                                      debug=debug, sample_method=sample_method)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")

refreasoning_dataroot = './data'
register_all_refreasoning_sg_counter(refreasoning_dataroot) # refreason_sg_counter
register_all_refreasoning_sg_counter(refreasoning_dataroot, debug=True) # refreason_sg_counter