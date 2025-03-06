# standard library imports
import io
import logging
import pathlib
import struct
from typing import Tuple

# 3rd party library imports
import imageio.v3 as iio

# local imports
from .jp2k import Jp2k
from .options import set_option
from ._core_converter import _2JP2Converter


class JPEG2JP2(_2JP2Converter):
    """
    Attributes
    ----------
    create_exif_uuid : bool
        Create a UUIDBox for the Exif metadata.  Always True for JPEG.
    jp2_filename : path
        Path to JPEG 2000 file to be written.
    jpeg_filename : path
        Path to JPEG file.
    tilesize : tuple
        The dimensions of a tile in the JP2K file.
    verbosity : int
        Set the level of logging, i.e. WARNING, INFO, etc.
    tags : dict
        Tags retrieved from APP1 segment, if any.
    """
    def __init__(
        self,
        jpeg: pathlib.Path | str,
        jp2: pathlib.Path | str,
        include_icc_profile: bool = False,
        num_threads: int = 1,
        tilesize: Tuple[int, int] | None = None,
        verbosity: int = logging.CRITICAL,
        **kwargs
    ):
        super().__init__(True, True, include_icc_profile, tilesize, verbosity)

        self.jpeg_path = pathlib.Path(jpeg)

        self.jp2_path = pathlib.Path(jp2)
        if self.jp2_path.exists():
            raise FileExistsError(f'{str(self.jp2_path)} already exists, please delete if you wish to overwrite.')  # noqa : E501

        self.jp2_kwargs = kwargs

        self.tags = None

        # This is never set for JPEG
        self.exclude_tags = None

        if num_threads > 1:
            set_option("lib.num_threads", num_threads)

    def __enter__(self):
        """The JPEG2JP2 object must be used with a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def run(self):

        self.copy_image()
        self.copy_metadata()

    def copy_metadata(self):
        """Transfer any EXIF or XMP metadata from the APPx segments."""

        with self.jpeg_path.open(mode='rb') as f:

            eof = False
            while not eof:

                marker = f.read(2)

                match marker:

                    case b'\xff\xd8':
                        # marker-only, SOI
                        pass

                    case b'\xff\xe0':
                        # APP0 JFIF, just skip over it
                        self.logger.warning('Ignoring APP0 JFIF segment...')
                        data = f.read(2)
                        size, = struct.unpack('>H', data)
                        buffer = f.read(size - 2)

                    case b'\xff\xe1':
                        # EXIF using APP1
                        data = f.read(2)
                        size, = struct.unpack('>H', data)
                        buffer = f.read(size - 2)

                        self.process_app1_segment(buffer)

                    case b'\xff\xe2':
                        # ICC profile
                        data = f.read(2)
                        size, = struct.unpack('>H', data)
                        buffer = f.read(size - 2)

                        self.process_app2_segment(buffer)

                    case b'\xff\xec':
                        # ducky?  ignore
                        data = f.read(2)
                        size, = struct.unpack('>H', data)
                        buffer = f.read(size - 2)

                    case b'\xff\xee':
                        # Adobe?
                        data = f.read(2)
                        size, = struct.unpack('>H', data)
                        buffer = f.read(size - 2)

                    case _:
                        # We don't care about anything else.  No need to scan
                        # the file any further, we're done.
                        eof = True

    def process_app1_segment(self, buffer):
        """
        An APP1 segment can contain Exif or XMP data.
        """

        if buffer[:6] == b'Exif\x00\x00':

            # ok it is Exif

            buffer = buffer[6:]

            bf = io.BytesIO(buffer)

            self.read_tiff_header(bf)
            self.tags = self.read_ifd(bf)
            self.append_exif_uuid_box()

        elif buffer[:28] == b'http://ns.adobe.com/xap/1.0/':

            # XMP APP segment
            self.xmp_data = buffer[29:]
            self.append_xmp_uuid_box()

        else:

            offset = f.tell() - 2 - 2 - size
            msg = f'Unrecognized APP1 segment at offset {offset}'
            self.logger.warning(msg)

    def process_app2_segment(self, buffer):
        """
        The APP2 segment(s) usually contains an ICC profile.  It may be split
        across more than one APP2 segment.
        """

        if buffer[:12] == b'ICC_PROFILE\x00':
            if not self.include_icc_profile:
                msg = 'ICC profile detected (skipped)'
                self.logger.warning(msg)
            count, nchunks = struct.unpack('BB', buffer[12:14])

            self.icc_profile = bytes(buffer[14:])
            self.rewrap_for_icc_profile()

    def copy_image(self):
        """Transfer the image data from the JPEG to the JP2 file."""
        image = iio.imread(self.jpeg_path)

        self.jp2 = Jp2k(
            self.jp2_path,
            tilesize=self.tilesize,
            **self.jp2_kwargs
        )

        self.jp2[:] = image
