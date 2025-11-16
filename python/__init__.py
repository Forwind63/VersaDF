"""
Introduction of MindRecord.

MindRecord is an efficient data storage and reading module provided by MindSpore.
This module provides several methods to help users convert various public datasets into the MindRecord format,
as well as methods to read, write, and retrieve data from MindRecord files.

.. image:: data_conversion_concept_en.png

MindSpore format data allows for more convenient saving and loading of data,
with the goal of normalizing user datasets and optimizing performance for different data scenarios.
Using the MindRecord data format can reduce disk I/O and network I/O overhead,
thereby providing a better data loading experience.

Users can generate MindRecord format data files using `mindspore.mindrecord.FileWriter` and load MindRecord format
datasets using `mindspore.dataset.MindDataset <https://www.mindspore.cn/docs/en/master/api_python/dataset/
mindspore.dataset.MindDataset.html>`_ .

Users can also convert datasets from other formats to the MindRecord format.
For more details, please refer to
`Converting Dataset to MindRecord <https://www.mindspore.cn/tutorials/en/master/dataset/record.html>`_ .
Additionally, MindRecord supports file encryption, decryption,
and integrity checks to ensure the security of MindRecord format datasets.
"""

from .filewriter import FileWriter
from .filereader import FileReader
from .mindpage import MindPage
from .common.exceptions import *
from .core.shardutils import SUCCESS, FAILED
from .tools.cifar10_to_mr import Cifar10ToMR
from .tools.cifar100_to_mr import Cifar100ToMR
from .tools.csv_to_mr import CsvToMR
from .tools.imagenet_to_mr import ImageNetToMR
from .tools.mnist_to_mr import MnistToMR
from .tools.tfrecord_to_mr import TFRecordToMR
from .config import *

__all__ = ['FileWriter', 'FileReader', 'MindPage',
           'Cifar10ToMR', 'Cifar100ToMR', 'CsvToMR', 'ImageNetToMR', 'MnistToMR', 'TFRecordToMR',
           'set_enc_key', 'set_enc_mode', 'set_dec_mode',
           'SUCCESS', 'FAILED']
