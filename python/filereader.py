"""
This module is to read data from versadf.
"""
import platform
from mindspore import log as logger

from .core.shardreader import ShardReader
from .core.shardheader import ShardHeader
from .core.shardutils import populate_data
from .core.shardutils import check_parameter
from .common.exceptions import ParamTypeError
from .config import _get_enc_key, _get_dec_mode, decrypt

__all__ = ['FileReader']


class FileReader:
    """
    Class to read versadf files.

    Note:
        If `file_name` is a file path, it tries to load all versadf files generated \
        in a conversion, and throws an exception if a versadf file is missing.
        If `file_name` is file path list, only the versadf files in the list are loaded.
        The parameter `operator` has no effect and will be deprecated in a future version.

    Args:
        file_name (str, list[str]): One of versadf file path or file path list.
        num_consumer (int, optional): Number of reader workers which load data. Default: ``4`` .
            It should not be smaller than 1 or larger than the number of processor cores.
        columns (list[str], optional): A list of fields where corresponding data would be read. Default: ``None`` .
        operator (int, optional): Reserved parameter for operators. Default: ``None`` .

    Raises:
        ParamValueError: If `file_name` , `num_consumer` or `columns` is invalid.

    Examples:
        >>> from mindspore.versadf import FileReader
        >>>
        >>> versadf_file = "/path/to/versadf/file"
        >>> reader = FileReader(file_name=versadf_file)
        >>>
        >>> # create iterator for versadf and get saved data
        >>> for _, item in enumerate(reader.get_next()):
        ...     ori_data = item
        >>> reader.close()
    """

    @check_parameter
    def __init__(self, file_name, num_consumer=4, columns=None, operator=None):
        if operator is not None:
            logger.warning("The parameter 'operator' will be deprecated in a future version.")

        if columns:
            if isinstance(columns, list):
                self._columns = columns
            else:
                raise ParamTypeError('columns', 'list')
        else:
            self._columns = None

        self._reader = ShardReader()
        self._file_name = ""
        if platform.system().lower() == "windows":
            if isinstance(file_name, list):
                self._file_name = [item.replace("\\", "/") for item in file_name]
            else:
                self._file_name = file_name.replace("\\", "/")
        else:
            self._file_name = file_name

        if isinstance(self._file_name, str):
            # decrypt the data file and index file
            index_file_name = self._file_name + ".db"
            decrypt_filename = decrypt(self._file_name, _get_enc_key(), _get_dec_mode())
            self._file_name = decrypt_filename
            decrypt(index_file_name, _get_enc_key(), _get_dec_mode())
        else:
            file_names_decrypted = []
            for item in self._file_name:
                # decrypt the data file and index file
                index_file_name = item + ".db"
                decrypt_filename = decrypt(item, _get_enc_key(), _get_dec_mode())
                file_names_decrypted.append(decrypt_filename)
                decrypt(index_file_name, _get_enc_key(), _get_dec_mode())
            self._file_name = file_names_decrypted

        self._reader.open(self._file_name, num_consumer, columns, operator)
        self._header = ShardHeader(self._reader.get_header())
        self._reader.launch()

    def get_next(self):
        """
        Yield a batch of data according to columns at a time.

        Note:
            Please refer to the Examples of :class:`mindspore.versadf.FileReader` .

        Returns:
            dict, a batch whose keys are the same as columns.

        Raises:
            MRMUnsupportedSchemaError: If schema is invalid.
        """
        iterator = self._reader.get_next()
        while iterator:
            for blob, raw in iterator:
                yield populate_data(raw, blob, self._columns, self._header.blob_fields, self._header.schema)
            iterator = self._reader.get_next()

    def close(self):
        """
        Stop reader worker and close file.

        Note:
            Please refer to the Examples of :class:`mindspore.versadf.FileReader` .
        """
        self._reader.close()

    def schema(self):
        """
        Get the schema of the versadf.

        Returns:
            dict, the schema info.

        Examples:
            >>> from mindspore.versadf import FileReader
            >>>
            >>> versadf_file = "/path/to/versadf/file"
            >>> reader = FileReader(file_name=versadf_file)
            >>> schema = reader.schema()
            >>> reader.close()
        """
        return self._header.schema

    def len(self):
        """
        Get the number of the samples in versadf.

        Returns:
            int, the number of the samples in versadf.

        Examples:
            >>> from mindspore.versadf import FileReader
            >>>
            >>> versadf_file = "/path/to/versadf/file"
            >>> reader = FileReader(file_name=versadf_file)
            >>> length = reader.len()
            >>> reader.close()
        """
        return self._reader.len()
