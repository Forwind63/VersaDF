"""
enums for exceptions
"""


class BaseEnum:
    """
    Enum base class.
    """


class LogRuntime(BaseEnum):
    """Log runtime enum."""
    RT_HOST = 0b01
    RT_DEVICE = 0b10


class ErrorCodeType(BaseEnum):
    """Error code type enum."""
    ERROR_CODE = 0b01
    EXCEPTION_CODE = 0b10


class ErrorLevel(BaseEnum):
    """Error level."""
    COMMON_LEVEL = 0b000
    SUGGESTION_LEVEL = 0b001
    MINOR_LEVEL = 0b010
    MAJOR_LEVEL = 0b011
    CRITICAL_LEVEL = 0b100
