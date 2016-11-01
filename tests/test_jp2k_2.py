"""
Tests for glymur that require openjp2
"""
# Standard library imports ...
import os
import struct
import tempfile
import unittest
import warnings

# Third party library imports ...
import numpy as np

# Local imports
import glymur
from glymur import Jp2k
from . import fixtures


@unittest.skipIf(glymur.version.openjpeg_version < '2.0.0',
                 "Not to be run until unless 2.0.1 or higher is present")
class TestSuite(unittest.TestCase):
    """Only to be run in 2.0+."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()
        self.jpxfile = glymur.data.jpxfile()

    def test_ignore_pclr_cmap_cdef_on_old_read(self):
        """
        The old "read" interface allowed for passing ignore_pclr_cmap_cdef
        to read a palette dataset "uninterpolated".
        """
        jpx = Jp2k(self.jpxfile)
        jpx.ignore_pclr_cmap_cdef = True
        expected = jpx[:]

        jpx2 = Jp2k(self.jpxfile)
        with warnings.catch_warnings():
            # Ignore a deprecation warning.
            warnings.simplefilter('ignore')
            actual = jpx2.read(ignore_pclr_cmap_cdef=True)

        np.testing.assert_array_equal(actual, expected)

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_grey_with_extra_component(self):
        """version 2.0 cannot write gray + extra"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128, 2), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data)
            self.assertEqual(j.box[2].box[0].height, 128)
            self.assertEqual(j.box[2].box[0].width, 128)
            self.assertEqual(j.box[2].box[0].num_components, 2)
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_rgb_with_extra_component(self):
        """v2.0+ should be able to write extra components"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128, 4), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data)
            self.assertEqual(j.box[2].box[0].height, 128)
            self.assertEqual(j.box[2].box[0].width, 128)
            self.assertEqual(j.box[2].box[0].num_components, 4)
            self.assertEqual(j.box[2].box[1].colorspace, glymur.core.SRGB)

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_openjpeg_library_message(self):
        """
        Verify the error message produced by the openjpeg library
        """
        # This will confirm that the error callback mechanism is working.
        with open(self.jp2file, 'rb') as f:
            data = f.read()

        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            # Codestream starts at byte 3323. SIZ marker at 3233.
            # COD marker at 3282.  Subsampling at 3276.
            offset = 3223
            tfile.write(data[0:offset + 52])

            # Make the DY bytes of the SIZ segment zero.  That means that
            # a subsampling factor is zero, which is illegal.
            tfile.write(b'\x00')
            tfile.write(data[offset + 53:offset + 55])
            tfile.write(b'\x00')
            tfile.write(data[offset + 57:offset + 59])
            tfile.write(b'\x00')

            tfile.write(data[offset + 59:])
            tfile.flush()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                j = Jp2k(tfile.name)
                with self.assertRaises((IOError, OSError)):
                    j[::2, ::2]

    def test_bad_area_parameter(self):
        """Should error out appropriately if given a bad area parameter."""
        j = Jp2k(self.jp2file)
        with self.assertRaises(IOError):
            # Start corner must be >= 0
            j[-1:1, -1:1]
        with self.assertRaises(IOError):
            # End corner must be > 0
            j[10:0, 10:0]
        with self.assertRaises(IOError):
            # End corner must be >= start corner
            j[10:8, 10:8]

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_unrecognized_jp2_clrspace(self):
        """We only allow RGB and GRAYSCALE.  Should error out with others"""
        data = np.zeros((128, 128, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data, colorspace='cmyk')

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_asoc_label_box(self):
        """Test asoc and label box"""
        # Construct a fake file with an asoc and a label box, as
        # OpenJPEG doesn't have such a file.
        data = Jp2k(self.jp2file)[::2, ::2]
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            Jp2k(tfile.name, data=data)

            with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile2:

                # Offset of the codestream is where we start.
                read_buffer = tfile.read(77)
                tfile2.write(read_buffer)

                # read the rest of the file, it's the codestream.
                codestream = tfile.read()

                # Write the asoc superbox.
                # Length = 36, id is 'asoc'.
                write_buffer = struct.pack('>I4s', int(56), b'asoc')
                tfile2.write(write_buffer)

                # Write the contained label box
                write_buffer = struct.pack('>I4s', int(13), b'lbl ')
                tfile2.write(write_buffer)
                tfile2.write('label'.encode())

                # Write the xml box
                # Length = 36, id is 'xml '.
                write_buffer = struct.pack('>I4s', int(35), b'xml ')
                tfile2.write(write_buffer)

                write_buffer = '<test>this is a test</test>'
                write_buffer = write_buffer.encode()
                tfile2.write(write_buffer)

                # Now append the codestream.
                tfile2.write(codestream)
                tfile2.flush()

                jasoc = Jp2k(tfile2.name)
                self.assertEqual(jasoc.box[3].box_id, 'asoc')
                self.assertEqual(jasoc.box[3].box[0].box_id, 'lbl ')
                self.assertEqual(jasoc.box[3].box[0].label, 'label')
                self.assertEqual(jasoc.box[3].box[1].box_id, 'xml ')
