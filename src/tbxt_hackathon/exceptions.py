"""Typed exceptions for tbxt_hackathon."""


class TBXTError(Exception):
    """Base class for tbxt_hackathon errors."""


class DataError(TBXTError):
    """Raised when input data is malformed or missing required fields."""


class FoldAssignmentError(TBXTError):
    """Raised when fold assignment fails (e.g. empty cluster)."""


class ModelError(TBXTError):
    """Raised when the chemeleon model or its adapters fail to build/train."""
