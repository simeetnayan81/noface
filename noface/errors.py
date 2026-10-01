class NofaceError(Exception):
    """A problem the user can fix. The CLI prints it without a traceback."""


class GalleryMismatch(NofaceError):
    """Video embeddings and the reference gallery came from different models."""
