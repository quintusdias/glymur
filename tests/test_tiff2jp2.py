# standard library imports
import warnings

# Local imports
from glymur import Jp2k, Tiff2Jp2
from . import fixtures


class TestSuite(fixtures.TestCommon):

    def test_smoke(self):
        """
        SCENARIO:  Convert TIFF file to JP2

        EXPECTED RESULT:  data matches
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with Tiff2Jp2(self.tiff_file, self.temp_jp2_filename) as j:
                j.run()

        actual = Jp2k(self.temp_jp2_filename)[:]

        self.assertEqual(actual.shape, (213, 234, 3))
