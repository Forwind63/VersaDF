"""
This module is to  write data into versadf.
"""
import mindspore._c_versadf as ms
from mindspore import log as logger
from mindspore.versadf.common.exceptions import MRMIndexGeneratorError, MRMGenerateIndexError

__all__ = ['ShardIndexGenerator']


class ShardIndexGenerator:
    """
    Wrapper class which is represent ShardIndexGenerator class in c++ module.

    The class would generate db files for accelerating reading.

    Args:
        path (str): Absolute path of versadf File.
        append (bool): If True, open existed versadf Files for appending, or create new versadf Files.

    Raises:
        MRMIndexGeneratorError: If failed to create index generator.
    """
    def __init__(self, path, append=False):
        self._generator = ms.ShardIndexGenerator(path, append)
        if not self._generator:
            logger.critical("Failed to create index generator.")
            raise MRMIndexGeneratorError

    def build(self):
        """
        Build index generator.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMGenerateIndexError: If failed to build index generator.
        """
        ret = self._generator.build()
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to build index generator.")
            raise MRMGenerateIndexError
        return ret

    def write_to_db(self):
        """
        Create index field in table for reading data.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMGenerateIndexError: If failed to write to database.
        """
        ret = self._generator.write_to_db()
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to write to database.")
            raise MRMGenerateIndexError
        return ret
