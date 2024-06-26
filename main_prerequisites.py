# -*- coding: utf-8 -*-
"""Functions to transform annotations and create train/test dataset."""

import sys
from prerequisites.data_config import DataConfig
from prerequisites.dataset_transformers import (
    VG200Transformer, VRDTransformer
)
from prerequisites.download_images import download_images

TRANSFORMERS = {
    'VG200': VG200Transformer,
    'VRD': VRDTransformer
}


def main(datasets):
    """Run the data preprocessing and creation pipeline."""
    _path = 'prerequisites/'
    download_images(_path, datasets)
    for dataset in datasets:
        print('Creating annotations for ' + dataset)
        TRANSFORMERS[dataset](DataConfig(_path, dataset)).transform()
    print('Done.')


if __name__ == "__main__":
    if any(sys.argv[1:]):
        main(sys.argv[1:])
    else:
        main(['VRD'])
