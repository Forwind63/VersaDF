"""
Introduction of versadf.

versadf is an efficient data storage and reading module provided by MindSpore.
This module provides several methods to help users convert various public datasets into the versadf format,
as well as methods to read, write, and retrieve data from versadf files.

.. image:: data_conversion_concept_en.png

MindSpore format data allows for more convenient saving and loading of data,
with the goal of normalizing user datasets and optimizing performance for different data scenarios.
Using the versadf data format can reduce disk I/O and network I/O overhead,
thereby providing a better data loading experience.

Users can generate versadf format data files using `mindspore.versadf.FileWriter` and load versadf format
datasets using `mindspore.dataset.MindDataset <https://www.mindspore.cn/docs/en/master/api_python/dataset/
mindspore.dataset.MindDataset.html>`_ .

Users can also convert datasets from other formats to the versadf format.
For more details, please refer to
`Converting Dataset to versadf <https://www.mindspore.cn/tutorials/en/master/dataset/record.html>`_ .
Additionally, versadf supports file encryption, decryption,
and integrity checks to ensure the security of versadf format datasets.
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
