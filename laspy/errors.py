"""All the custom exceptions types"""


class LaspyException(Exception):
    pass


class UnknownExtraType(LaspyException):
    pass


class PointFormatNotSupported(LaspyException):
    pass


class FileVersionNotSupported(LaspyException):
    pass


class LazError(LaspyException):
    pass


class IncompatibleDataFormat(LaspyException):
    pass


class TruncatedPointDataError(LaspyException):
    """The point data ends before the header's declared point count.

    Raised by :meth:`laspy.LasReader.read_points` when the source yields fewer point
    records than the header promises, which means the file was truncated by a cut-short
    export or transfer. Upstream laspy only logs this and returns a short record, so the
    caller silently processes a partial file; this fork raises instead. Pass
    ``strict_point_count=False`` to :func:`laspy.open` for the old permissive behaviour.

    Keep this constructible from its message alone: exceptions cross Ray object
    boundaries by pickle, which reconstructs them as ``cls(*self.args)`` and then
    restores ``__dict__``. A signature that demands the extra arguments would arrive at
    the other end as an unserializable stand-in instead of this error.
    """

    def __init__(self, message, declared_point_count=None, points_present=None):
        super().__init__(message)
        self.declared_point_count = declared_point_count
        self.points_present = points_present
