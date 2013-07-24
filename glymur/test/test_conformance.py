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
    format_corpus_data_root = os.environ['FORMAT_CORPUS_DATA_ROOT']
except KeyError:
    format_corpus_data_root = None
except:
    raise

try:
    opj_data_root = os.environ['OPJ_DATA_ROOT']
except KeyError:
    opj_data_root = None
except:
    raise


@unittest.skipIf(format_corpus_data_root is None,
                 "FORMAT_CORPUS_DATA_ROOT environment variable not set")
@unittest.skipIf(sys.hexversion < 0x03020000,
                 "Requires features introduced in 3.2 (assertWarns)")
class TestSuiteFormatCorpus(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_balloon_trunc1(self):
        # Has one byte shaved off of EOC marker.
        jfile = os.path.join(format_corpus_data_root,
                             'jp2k-test/byteCorruption/balloon_trunc1.jp2')
        j2k = Jp2k(jfile)
        with self.assertWarns(UserWarning):
            c = j2k.get_codestream(header_only=False)
        
        # The last segment is truncated, so there should not be an EOC marker.
        self.assertNotEqual(c.segment[-1].marker_id, 'EOC')

        # The codestream is not as long as claimed.
        with self.assertRaises(OSError):
            j2k.read(rlevel=-1)

    def test_balloon_trunc2(self):
        # Shortened by 5000 bytes.
        jfile = os.path.join(format_corpus_data_root,
                             'jp2k-test/byteCorruption/balloon_trunc2.jp2')
        j2k = Jp2k(jfile)
        with self.assertWarns(UserWarning):
            c = j2k.get_codestream(header_only=False)
        
        # The last segment is truncated, so there should not be an EOC marker.
        self.assertNotEqual(c.segment[-1].marker_id, 'EOC')

        # The codestream is not as long as claimed.
        with self.assertRaises(OSError):
            j2k.read(rlevel=-1)

    def test_balloon_trunc3(self):
        # Most of last tile is missing.
        jfile = os.path.join(format_corpus_data_root,
                             'jp2k-test/byteCorruption/balloon_trunc3.jp2')
        j2k = Jp2k(jfile)
        with self.assertWarns(UserWarning):
            c = j2k.get_codestream(header_only=False)
        
        # The last segment is truncated, so there should not be an EOC marker.
        self.assertNotEqual(c.segment[-1].marker_id, 'EOC')

        # Should error out, it does not.
        #with self.assertRaises(OSError):
        #    j2k.read(rlevel=-1)

    def test_jp2_brand_vs_any_icc_profile(self):
        # If 'jp2 ', then the method cannot be any icc profile.
        jfile = os.path.join(format_corpus_data_root,
                             'jp2k-test', 'icc',
                             'balloon_eciRGBv2_ps_adobeplugin.jpf')
        with self.assertWarns(UserWarning):
            j2k = Jp2k(jfile)
        
    def test_jp2_brand_vs_any_icc_profile_multiple_colr(self):
        # Has colr box, one that conforms, one that does not.

        # Wrong 'brand' field; contains two versions of ICC profile: one
        # embedded using "Any ICC" method; other embedded using "Restricted
        # ICC" method, with description ("Modified eciRGB v2") and profileClass
        # ("Input Device") changed relative to original profile.
        lst = [format_corpus_data_root, 'jp2k-test', 'icc',
               'balloon_eciRGBv2_ps_adobeplugin_jp2compatible.jpf']
        jfile = os.path.join(*lst)
        with self.assertWarns(UserWarning):
            j2k = Jp2k(jfile)
        

@unittest.skipIf(opj_data_root is None,
                 "OPJ_DATA_ROOT environment variable not set")
@unittest.skipIf(sys.hexversion < 0x03020000,
                 "Requires features introduced in 3.2 (assertWarns)")
class TestSuiteOpj(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_jp2_brand_vs_any_icc_profile(self):
        # If 'jp2 ', then the method cannot be any icc profile.
        filename = os.path.join(opj_data_root,
                                'input/nonregression/text_GBR.jp2')
        with self.assertWarns(UserWarning):
            j2k = Jp2k(filename)
        
if __name__ == "__main__":
    unittest.main()
