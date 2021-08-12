# standard library imports
import shlex
import subprocess
import tempfile
import unittest

# 3rd party library imports
import numpy as np
import skimage.io

# local imports
import glymur
from glymur import Jp2k

class TestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.jp2 = Jp2k(glymur.data.nemo())

    def get_openjpeg_data(self, components):

        with tempfile.NamedTemporaryFile(suffix='.tif') as t:
            command = (
                f"opj_decompress "
                f"-i {str(self.jp2.filename)} "
                f"-o {t.name} "
                f"-c {components}"
            )
            args = shlex.split(command)
            p = subprocess.Popen(command, shell=True)
            p.wait()

            if p.returncode > 0:
                stdout, stderr = p.communicate()
                self.fail(stderr.decode('utf-8'))

            data = skimage.io.imread(t.name)
            return data

    def test_one_component(self):

        self.jp2.set_decoded_components(0)
        actual = self.jp2[:]

        expected = self.get_openjpeg_data([0])

        np.testing.assert_array_equal(actual, expected)
