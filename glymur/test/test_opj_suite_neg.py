"""
The tests here do not correspond directly to the OpenJPEG test suite, but
seem like logical negative tests to add.
"""
#pylint:  disable-all
import os
import sys
import tempfile

if sys.hexversion < 0x02070000:
    import unittest2 as unittest
else:
    import unittest

import numpy as np
import pkg_resources

from glymur.lib import openjp2 as opj2

from .fixtures import read_image

msg = "Matplotlib with the PIL backend must be available in order to run the "
msg += "tests in this suite."
no_read_backend_msg = msg
try:
    from PIL import Image
    from matplotlib.pyplot import imread
    no_read_backend = False
except:
    no_read_backend = True

from glymur import Jp2k
import glymur

try:
    data_root = os.environ['OPJ_DATA_ROOT']
except KeyError:
    data_root = None
except:
    raise


@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None,
                 "Missing openjp2 library.")
@unittest.skipIf(no_read_backend, no_read_backend_msg)
@unittest.skipIf(data_root is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuiteNegative(unittest.TestCase):

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_negative_psnr_with_cratios(self):
        # Using psnr with cratios options is not allowed.
        # Not an OpenJPEG test, but close.
        infile = os.path.join(data_root, 'input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            with self.assertRaises(IOError):
                j.write(data, psnr=[30, 35, 40], cratios=[2, 3, 4])

    def test_NR_MarkerIsNotCompliant_j2k_dump(self):
        relpath = 'input/nonregression/MarkerIsNotCompliant.j2k'
        jfile = os.path.join(data_root, relpath)
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_illegalcolortransform_dump(self):
        # EOC marker is bad
        relpath = 'input/nonregression/illegalcolortransform.j2k'
        jfile = os.path.join(data_root, relpath)
        jp2k = Jp2k(jfile)
        with self.assertWarns(UserWarning) as cw:
            c = jp2k.get_codestream(header_only=False)

        # Verify that the last segment returned in the codestream is SOD,
        # not EOC.  Codestream parsing should stop when we try to jump to
        # the end of SOT.
        self.assertEqual(c.segment[-1].marker_id, 'SOD')

    def test_NR_Cannotreaddatawithnosizeknown_j2k(self):
        relpath = 'input/nonregression/Cannotreaddatawithnosizeknown.j2k'
        jfile = os.path.join(data_root, relpath)
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_code_block_dimensions(self):
        # opj_compress doesn't allow the dimensions of a codeblock
        # to be too small or too big, so neither will we.
        data = np.zeros((256, 256), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')

            # opj_compress doesn't allow code block area to exceed 4096.
            with self.assertRaises(IOError) as cr:
                j.write(data, cbsize=(256, 256))

            # opj_compress doesn't allow either dimension to be less than 4.
            with self.assertRaises(IOError) as cr:
                j.write(data, cbsize=(2048, 2))
            with self.assertRaises(IOError) as cr:
                j.write(data, cbsize=(2, 2048))

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_exceeded_box(self):
        # Verify that a warning is issued if we read past the end of a box
        # This file has a palette (pclr) box whose length is impossibly
        # short.
        infile = os.path.join(data_root,
                              'input/nonregression/mem-b2ace68c-1381.jp2')
        with self.assertWarns(UserWarning) as cw:
            j = Jp2k(infile)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_precinct_size_not_multiple_of_two(self):
        # Seems like precinct sizes should be powers of two.
        ifile = Jp2k(self.j2kfile)
        data = ifile.read(rlevel=2)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            ofile = Jp2k(tfile.name, 'wb')
            with self.assertRaises(IOError) as ce:
                ofile.write(data, psizes=[(13, 13)])

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_codeblock_size_not_multiple_of_two(self):
        # Seems like code block sizes should be powers of two.
        ifile = Jp2k(self.j2kfile)
        data = ifile.read(rlevel=2)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            ofile = Jp2k(tfile.name, 'wb')
            with self.assertRaises(IOError) as ce:
                ofile.write(data, cbsize=(13, 12))

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_codeblock_size_with_precinct_size(self):
        # Seems like code block sizes should never exceed half that of
        # precinct size.
        ifile = Jp2k(self.j2kfile)
        data = ifile.read(rlevel=2)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            ofile = Jp2k(tfile.name, 'wb')
            with self.assertRaises(IOError) as ce:
                ofile.write(data,
                            cbsize=(64, 64),
                            psizes=[(64, 64)])

if __name__ == "__main__":
    unittest.main()
