# standard library imports
import importlib.resources as ir
import pathlib
import shutil
import tempfile
import unittest

# 3rd party library imports
import numpy as np

# local imports
import glymur
from glymur import Jp2k
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """
        We need a test image without the MCT and with at least 2 quality
        layers.
        """
        self.testdir = tempfile.mkdtemp()

        # windows won't like it if we try using tempfile.NamedTemporaryFile
        # here
        self.j2kfile = pathlib.Path(self.testdir) / 'tmp.j2k'

        data = Jp2k(glymur.data.goodstuff())[:]
        Jp2k(self.j2kfile.name, data=data, mct=False, cratios=[200, 100, 50])

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.testdir)

    def test_one_component(self):
        """
        SCENARIO:  Decode the 1st component of an RGB image.  Then restore
        the original configuration of reading all bands.

        EXPECTED RESULT:  the data matches what we get from the regular way.
        """
        j2k = Jp2k(self.j2kfile.name)
        expected = j2k[:, :, 0]

        j2k.decoded_components = 0
        actual = j2k[:]

        np.testing.assert_array_equal(actual, expected)

        # restore the original configuration
        j2k.decoded_components = None
        actual = j2k[:]
        self.assertEqual(actual.shape, (800, 480, 3))

    def test_second_component(self):
        """
        SCENARIO:  Decode the 2nd component of a non-MCT image.

        EXPECTED RESULT:  Match the 2nd component read in the regular way.
        """
        j2k = Jp2k(self.j2kfile.name)
        expected = j2k[:, :, 1]

        j2k.decoded_components = 1
        actual = j2k[:]

        np.testing.assert_array_equal(actual, expected)

    def test_three_components_without_MCT(self):
        """
        SCENARIO:  Decode three components without using the MCT.

        EXPECTED RESULT:  Match the results the regular way.
        """

        j2k = Jp2k(self.j2kfile.name)

        expected = j2k[:]

        j2k.decoded_components = [0, 1, 2]
        actual = j2k[:]

        np.testing.assert_array_equal(actual, expected)

    def test_partial_component_decoding_with_area(self):
        """
        SCENARIO:  Decode one component with a specific area.

        EXPECTED RESULT:  Match the results the regular way.
        """
        j2k = Jp2k(self.j2kfile.name)

        expected = j2k[20:40, 10:30, 0]

        j2k.decoded_components = 0
        actual = j2k[20:40, 10:30]

        np.testing.assert_array_equal(actual, expected)

    def test_layer(self):
        """
        SCENARIO:  Decode one component with a particular layer

        EXPECTED RESULT:  Match the results the regular way.
        """

        j2k = Jp2k(self.j2kfile.name)
        j2k.layer = 1

        expected = j2k[:, :, 0]

        j2k.decoded_components = 0
        actual = j2k[:]

        np.testing.assert_array_equal(actual, expected)

    def test_reduced_resolution(self):
        """
        SCENARIO:  Decode one component with reduced resolution.

        EXPECTED RESULT:  Match the results the regular way.
        """

        j2k = Jp2k(self.j2kfile.name)

        expected = j2k[::2, ::2, 0]

        j2k.decoded_components = [0]
        actual = j2k[::2, ::2]

        np.testing.assert_array_equal(actual, expected)

    def test_negative_component(self):
        """
        SCENARIO:  Provide a negative component.

        EXPECTED RESULT:  exception
        """

        j2k = Jp2k(self.j2kfile.name)

        with self.assertRaises(glymur.lib.openjp2.OpenJPEGLibraryError):
            j2k.decoded_components = -1
            j2k[:]

    def test_same_component_several_times(self):
        """
        SCENARIO:  Decode one component multiple times.

        EXPECTED RESULT:  exception
        """

        j2k = Jp2k(self.j2kfile.name)

        with self.assertRaises(glymur.lib.openjp2.OpenJPEGLibraryError):
            j2k.decoded_components = [0, 0]
            j2k[:]

    def test_invalid_component(self):
        """
        SCENARIO:  Decode an invalid component.

        EXPECTED RESULT:  exception
        """
        j2k = Jp2k(self.j2kfile.name)

        with self.assertRaises(ValueError):
            j2k.decoded_components = 10

    def test_differing_subsamples(self):
        """
        SCENARIO:  Decode a component where other components have different
        subsamples.

        EXPECTED RESULT:  success, trying to read that component without
        setting decoded_components would require us to use the read_bands
        method.
        """
        with ir.path('tests.data', 'p0_06.j2k') as path:
            j2k = Jp2k(path)

            expected = j2k.read_bands()[0]

            j2k.decoded_components = 0
            actual = j2k[:]

            np.testing.assert_array_equal(actual, expected)

        # verify that without using decoded components, we cannot read the
        # image using the slice protocol
        with ir.path('tests.data', 'p0_06.j2k') as path:
            j2k = Jp2k(path)
            with self.assertRaises(RuntimeError):
                j2k[:, :, 0]
