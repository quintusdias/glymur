# standard library imports
import pathlib
from typing import Tuple

# 3rd party library imports
import skimage

# local imports
from .jp2k import Jp2k


class JPEG2JP2(object):
    """
    Attributes
    ----------
    jp2_filename : path
        Path to JPEG 2000 file to be written.
    jpeg_filename : path
        Path to JPEG file.
    tilesize : tuple
        The dimensions of a tile in the JP2K file.
    """
    def __init__(
        self,
        jpeg: pathlib.Path,
        jp2: pathlib.Path,
        tilesize: Tuple[int, int] | None = None,
        **kwargs
    ):
        self.jpeg = jpeg
        self.jp2 = jp2
        self.tilesize = tilesize

    def __enter__(self):
        """The JPEG2JP2 object must be used with a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def run(self):

        self.copy_image()

    def copy_image(self):
        """Transfer the image data from the JPEG to the JP2 file."""
        image = skimage.io.imread(self.jpeg)

        self.jp2 = Jp2k(
            self.jp2,
            tilesize=self.tilesize,
        )

        self.jp2[:] = image
