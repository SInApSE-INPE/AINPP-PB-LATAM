from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ainpp-pb-latam")
except PackageNotFoundError:
    __version__ = "unknown"