import os
import struct
import sys
import tempfile
import unittest

import numpy as np
import pkg_resources

from glymur import Jp2k
import glymur

try:
    data_root = os.environ['OPJ_DATA_ROOT']
except KeyError:
    data_root = None
except:
    raise


@unittest.skipIf(data_root is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestCodestream(unittest.TestCase):

    def setUp(self):
        self.jp2file = pkg_resources.resource_filename(glymur.__name__,
                                                       "data/nemo.jp2")

    def tearDown(self):
        pass

    def test_reserved_marker_segment(self):
        # Some marker segments were reserved in FCD15444-1.  Since that
        # standard is old, some of them may have come into use.
        #
        # Let's inject a reserved marker segment into a file that
        # we know something about to make sure we can still parse it.
        filename = os.path.join(data_root, 'input/conformance/p0_01.j2k')
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with open(filename, 'rb') as ifile:
                # Everything up until the first QCD marker.
                buffer = ifile.read(45)
                tfile.write(buffer)

                # Write the new marker segment, 0xff6f = 65391
                buffer = struct.pack('>HHB', int(65391), int(3), int(0))
                tfile.write(buffer)

                # Get the rest of the input file.
                buffer = ifile.read()
                tfile.write(buffer)
                tfile.flush()

            j = Jp2k(tfile.name)
            c = j.get_codestream()

            self.assertEqual(c.segment[2].id, '0xff6f')
            self.assertEqual(c.segment[2].length, 3)
            self.assertEqual(c.segment[2].data, b'\x00')

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_unknown_marker_segment(self):
        # Let's inject a marker segment whose marker does not appear to
        # be valid.  We still parse the file, but warn about the offending
        # marker.
        filename = os.path.join(data_root, 'input/conformance/p0_01.j2k')
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with open(filename, 'rb') as ifile:
                # Everything up until the first QCD marker.
                buffer = ifile.read(45)
                tfile.write(buffer)

                # Write the new marker segment, 0xff79 = 65401
                buffer = struct.pack('>HHB', int(65401), int(3), int(0))
                tfile.write(buffer)

                # Get the rest of the input file.
                buffer = ifile.read()
                tfile.write(buffer)
                tfile.flush()

            with self.assertWarns(UserWarning) as cw:
                j = Jp2k(tfile.name)
                c = j.get_codestream()

            self.assertEqual(c.segment[2].id, '0xff79')
            self.assertEqual(c.segment[2].length, 3)
            self.assertEqual(c.segment[2].data, b'\x00')

if __name__ == "__main__":
    unittest.main()
