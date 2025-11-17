"""
This module is to read data from versadf.
"""
import mindspore._c_versadf as ms
from mindspore import log as logger
from mindspore.versadf.common.exceptions import MRMOpenError, MRMLaunchError

__all__ = ['ShardReader']


class ShardReader:
    """
    Wrapper class which is represent ShardReader class in c++ module.

    The class would read a batch of data from versadf File series.
    """
    def __init__(self):
        self._reader = ms.ShardReader()

    def open(self, file_name, num_consumer=4, columns=None, operator=None):
        """
        Open file and prepare to read versadf File.

        Args:
           file_name (str, list[str]): File names of versadf File.
           num_consumer (int): Number of worker threads which load data in parallel. Default: 4.
           columns (list[str]): List of fields which correspond data would be read.
           operator(int): Reserved parameter for operators. Default: ``None``.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMOpenError: If failed to open versadf File.
        """
        columns = columns if columns else []
        operator = operator if operator else []
        if isinstance(file_name, list):
            load_dataset = False
        else:
            load_dataset = True
            file_name = [file_name]
        ret = self._reader.open(file_name, load_dataset, num_consumer, columns, operator)
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to open {}.".format(file_name))
            self.close()
            raise MRMOpenError
        return ret

    def launch(self):
        """
        Launch the worker threads to load data.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMLaunchError: If failed to launch worker threads.
        """
        ret = self._reader.launch()
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to launch worker threads.")
            raise MRMLaunchError
        return ret

    def get_next(self):
        """
        Return a batch of data including blob data and raw data.

        Returns:
           list of dict.
        """
        return self._reader.get_next()

    def get_blob_fields(self):
        """
        Return blob fields of versadf.

        Returns:
            list of str.
        """
        return self._reader.get_blob_fields()

    def get_header(self):
        """
        Return header of versadf.

        Returns:
            pointer object refer to header.
        """
        return self._reader.get_header()

    def close(self):
        """close versadf File."""
        self._reader.close()

    def len(self):
        """
        Get the number of the samples in versadf.

        Returns:
            int, the number of the samples in versadf.
        """
        return self._reader.len()
