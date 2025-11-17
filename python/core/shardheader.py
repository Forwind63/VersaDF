"""
This module is to write data into versadf.
"""
import mindspore._c_versadf as ms
from mindspore import log as logger
from mindspore.versadf.common.exceptions import MRMAddSchemaError, MRMAddIndexError, MRMBuildSchemaError, \
    MRMGetMetaError

__all__ = ['ShardHeader']


class ShardHeader:
    """
    Wrapper class which is represent ShardHeader class in c++ module.

    The class would store meta data of versadf File.
    """
    def __init__(self, header=None):
        if header:
            self._header = header
        else:
            self._header = ms.ShardHeader()

    @property
    def header(self):
        """Getter of header"""
        return self._header

    @property
    def blob_fields(self):
        """Getter of blob fields"""
        return self._get_blob_fields()

    @property
    def schema(self):
        """Getter of schema"""
        return self._get_schema()

    def add_schema(self, schema):
        """
        Add object of ShardSchema.

        Args:
          schema (ShardSchema): Object of ShardSchema.

        Returns:
           int, schema id.

        Raises:
            MRMAddSchemaError: If failed to add schema.
        """
        schema_id = self._header.add_schema(schema)
        if schema_id == -1:
            logger.critical("Failed to add schema.")
            raise MRMAddSchemaError
        return schema_id

    def add_index_fields(self, index_fields):
        """
        Add object of ShardSchema.

        Args:
          index_fields (list[str]):

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMAddSchemaError: If failed to add index field.
        """
        ret = self._header.add_index_fields(index_fields)
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to add index field.")
            raise MRMAddIndexError
        return ret

    def build_schema(self, content, desc=None):
        """
        Build raw schema to generate schema object.

        Args:
            content (dict): Dict of user defined schema.
            desc (str,optional): String of schema description.

        Returns:
           Class ShardSchema.

        Raises:
            MRMBuildSchemaError: If failed to build schema.
        """
        desc = desc if desc else ""
        schema = ms.Schema.build(desc, content)
        if not schema:
            logger.critical("Failed to add build schema.")
            raise MRMBuildSchemaError
        return schema

    def _get_schema(self):
        """
        Get schema info.

        Returns:
             List of dict.
        """
        return self._get_meta()['schema']

    def _get_blob_fields(self):
        """
        Get blob fields info.

        Returns:
             List of dict.
        """
        return self._get_meta()['blob_fields']

    def _get_meta(self):
        """
       Get metadata including schema, blob fields .etc.

       Returns:
            List of dict.
       """
        ret = self._header.get_meta()
        if ret and len(ret) == 1:
            return ret[0].get_schema_content()

        logger.critical("Failed to get meta info.")
        raise MRMGetMetaError
