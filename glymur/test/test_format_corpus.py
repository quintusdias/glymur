"""
These tests deal with JPX/JP2/J2K images in the format-corpus repository.
"""
#pylint:  disable-all

import os
import sys

if sys.hexversion < 0x02070000:
    import unittest2 as unittest
else:
    import unittest

import warnings

from glymur import Jp2k
import glymur

try:
    data_root = os.environ['FORMAT_CORPUS_ROOT']
except KeyError:
    data_root = None
except:
    raise


@unittest.skipIf(sys.hexversion < 0x03020000,
                 "Requires features introduced in 3.2 (assertWarns)")
@unittest.skipIf(data_root is None,
                 "FORMAT_CORPUS_ROOT environment variable not set")
class TestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_balloon_trunc1(self):
        # Has one byte shaved off of EOC marker.
        jfile = os.path.join(data_root,
                             'jp2k-test/byteCorruption/balloon_trunc1.jp2')
        j2k = Jp2k(jfile)
        with self.assertWarns(UserWarning):
            c = j2k.get_codestream(header_only=False)
        
        # The last segment is truncated, so there should not be an EOC marker.
        self.assertNotEqual(c.segment[-1].marker_id, 'EOC')

if __name__ == "__main__":
    unittest.main()
