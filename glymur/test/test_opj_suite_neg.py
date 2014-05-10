"""
The tests here do not correspond directly to the OpenJPEG test suite, but
seem like logical negative tests to add.
"""
# E1101:  assertWarns introduced in python 3.2
# pylint: disable=E1101

# R0904:  Not too many methods in unittest.
# pylint: disable=R0904

import os
import re
import sys
import tempfile
import unittest
import warnings

import numpy as np

try:
    import skimage.io
    skimage.io.use_plugin('freeimage', 'imread')
    _HAS_SKIMAGE_FREEIMAGE_SUPPORT = True
except ((ImportError, RuntimeError)):
    _HAS_SKIMAGE_FREEIMAGE_SUPPORT = False

from .fixtures import OPJ_DATA_ROOT, opj_data_file, read_image
from .fixtures import NO_READ_BACKEND, NO_READ_BACKEND_MSG

from glymur import Jp2k
import glymur


@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_OPJ_DATA_ROOT environment variable not set")
class TestSuiteNegative(unittest.TestCase):
    """Test suite for certain negative tests from openjpeg suite."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass


    @unittest.skipIf(not _HAS_SKIMAGE_FREEIMAGE_SUPPORT,
                     "Cannot read input image without scikit-image/freeimage")
    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_cinema2K_bad_frame_rate(self):
        """Cinema2k frame rate must be either 24 or 48."""
        relfile = 'input/nonregression/X_5_2K_24_235_CBR_STEM24_000.tif'
        infile = opj_data_file(relfile)
        data = skimage.io.imread(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            with self.assertRaises(IOError):
                j.write(data, cinema2k=36)


    @unittest.skipIf(NO_READ_BACKEND, NO_READ_BACKEND_MSG)
    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_psnr_with_cratios(self):
        """Using psnr with cratios options is not allowed."""
        # Not an OpenJPEG test, but close.
        infile = opj_data_file('input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            with self.assertRaises(IOError):
                j.write(data, psnr=[30, 35, 40], cratios=[2, 3, 4])

    def test_nr_marker_not_compliant(self):
        """non-compliant marker, should still be able to read"""
        relpath = 'input/nonregression/MarkerIsNotCompliant.j2k'
        jfile = opj_data_file(relpath)
        jp2k = Jp2k(jfile)
        jp2k.get_codestream(header_only=False)
        self.assertTrue(True)

    def test_nr_illegalclrtransform(self):
        """EOC marker is bad"""
        relpath = 'input/nonregression/illegalcolortransform.j2k'
        jfile = opj_data_file(relpath)
        jp2k = Jp2k(jfile)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            codestream = jp2k.get_codestream(header_only=False)

        # Verify that the last segment returned in the codestream is SOD,
        # not EOC.  Codestream parsing should stop when we try to jump to
        # the end of SOT.
        self.assertEqual(codestream.segment[-1].marker_id, 'SOD')

    def test_nr_cannotreadwnosizeknown(self):
        """not sure exactly what is wrong with this file"""
        relpath = 'input/nonregression/Cannotreaddatawithnosizeknown.j2k'
        jfile = opj_data_file(relpath)
        jp2k = Jp2k(jfile)
        jp2k.get_codestream(header_only=False)
        self.assertTrue(True)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_code_block_dimensions(self):
        """don't allow extreme codeblock sizes"""
        # opj_compress doesn't allow the dimensions of a codeblock
        # to be too small or too big, so neither will we.
        data = np.zeros((256, 256), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')

            # opj_compress doesn't allow code block area to exceed 4096.
            with self.assertRaises(IOError):
                j.write(data, cbsize=(256, 256))

            # opj_compress doesn't allow either dimension to be less than 4.
            with self.assertRaises(IOError):
                j.write(data, cbsize=(2048, 2))
            with self.assertRaises(IOError):
                j.write(data, cbsize=(2, 2048))

    def test_exceeded_box(self):
        """should warn if reading past end of a box"""
        # Verify that a warning is issued if we read past the end of a box
        # This file has a palette (pclr) box whose length is impossibly
        # short.
        infile = os.path.join(OPJ_DATA_ROOT,
                              'input/nonregression/mem-b2ace68c-1381.jp2')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            Jp2k(infile)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_precinct_size_not_p2(self):
        """precinct sizes should be powers of two."""
        ifile = Jp2k(self.j2kfile)
        data = ifile.read(rlevel=2)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            ofile = Jp2k(tfile.name, 'wb')
            with self.assertRaises(IOError):
                ofile.write(data, psizes=[(13, 13)])

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_cblk_size_not_power_of_two(self):
        """code block sizes should be powers of two."""
        ifile = Jp2k(self.j2kfile)
        data = ifile.read(rlevel=2)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            ofile = Jp2k(tfile.name, 'wb')
            with self.assertRaises(IOError):
                ofile.write(data, cbsize=(13, 12))

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_cblk_size_precinct_size(self):
        """code block sizes should never exceed half that of precinct size."""
        ifile = Jp2k(self.j2kfile)
        data = ifile.read(rlevel=2)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            ofile = Jp2k(tfile.name, 'wb')
            with self.assertRaises(IOError):
                ofile.write(data,
                            cbsize=(64, 64),
                            psizes=[(64, 64)])

if __name__ == "__main__":
    unittest.main()
