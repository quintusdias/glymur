# standard library imports
import os
import pathlib
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import warnings

# 3rd party library imports
import numpy as np

# local imports
import glymur
from glymur import Jp2k
from glymur.jp2box import InvalidJp2kError
from . import fixtures
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


@unittest.skipIf(
    not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
)
@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(fixtures.TestCommon):
    """Test writing Jpeg2000 files"""

    @classmethod
    def setUpClass(cls):
        cls.jp2file = glymur.data.nemo()
        cls.j2kfile = glymur.data.goodstuff()

        cls.j2k_data = glymur.Jp2k(cls.j2kfile)[:]
        cls.jp2_data = glymur.Jp2k(cls.jp2file)[:]

        # Make single channel jp2 and j2k files.
        test_dir = tempfile.mkdtemp()
        test_dir_path = pathlib.Path(test_dir)

        cls.single_channel_j2k = test_dir_path / 'single_channel.j2k'
        glymur.Jp2k(cls.single_channel_j2k, data=cls.j2k_data[:, :, 0])

        cls.single_channel_jp2 = test_dir_path / 'single_channel.jp2'
        glymur.Jp2k(cls.single_channel_jp2, data=cls.j2k_data[:, :, 0])

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.single_channel_j2k)
        os.unlink(cls.single_channel_jp2)

    def _verify_size_segment(self, actual, expected):
        """
        Verify the fields of the SIZ segment.
        """
        for field in [
            'rsiz', 'xsiz', 'ysiz', 'xosiz', 'yosiz', 'xtsiz', 'ytsiz',
            'xtosiz', 'ytosiz', 'bitdepth', 'xrsiz', 'yrsiz'
        ]:
            self.assertEqual(getattr(actual, field), getattr(expected, field))

    def _verify_codeblock_style(self, actual, styles):
        """
        Verify the code-block style for SPcod and SPcoc parameters.

        This information is stored in a single byte.  Please reference
        Table A-17 in FCD15444-1
        """
        expected = 0
        masks = [
            0x01,  # Selective arithmetic coding bypass
            0x02,  # Reset context probabilities
            0x04,  # Termination on each coding pass
            0x08,  # Vertically causal context
            0x10,  # Predictable termination
            0x20,  # Segmentation symbols
        ]
        for style, mask in zip(styles, masks):
            if style:
                expected |= mask

        self.assertEqual(actual, expected)

    def test_write_to_fully_formed_jp2k(self):
        """
        Scenario:  Attempt to write to a fully formed file.

        Expected Result:  RuntimeError
        """
        j = Jp2k(self.temp_jp2_filename, data=self.jp2_data)
        with self.assertRaises(RuntimeError):
            j[:] = np.ones((100, 100), dtype=np.uint8)

    def test_capture_resolution(self):
        """
        SCENARIO:  The capture_resolution keyword is specified.

        EXPECTED RESULT:  The cres box is created.
        """
        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4
        j = glymur.Jp2k(
            self.temp_jp2_filename, data=self.jp2_data,
            capture_resolution=[vresc, hresc],
            display_resolution=[vresd, hresd],
        )

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resc')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresc)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresc)

        self.assertEqual(j.box[2].box[2].box[1].box_id, 'resd')
        self.assertEqual(j.box[2].box[2].box[1].vertical_resolution, vresd)
        self.assertEqual(j.box[2].box[2].box[1].horizontal_resolution, hresd)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_capture_resolution_camera(self):
        """
        SCENARIO:  The capture_resolution keyword is specified.

        EXPECTED RESULT:  The offset and length of the resolution superbox
        are verified.
        """
        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4
        j = glymur.Jp2k(
            self.temp_jp2_filename, data=fixtures.skimage.data.camera(),
            capture_resolution=[vresc, hresc],
            display_resolution=[vresd, hresd],
        )

        self.assertEqual(j.box[2].box[2].box_id, 'res ')
        self.assertEqual(j.box[2].box[2].length, 44)
        self.assertEqual(j.box[2].box[2].offset, 77)

    def test_capture_resolution_when_j2k_specified(self):
        """
        Scenario:  Capture/Display resolution boxes are specified when the file
        name indicates J2K.

        Expected Result:  InvalidJp2kError
        """

        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4
        with self.assertRaises(InvalidJp2kError):
            glymur.Jp2k(
                self.temp_j2k_filename, data=self.jp2_data,
                capture_resolution=[vresc, hresc],
                display_resolution=[vresd, hresd],
            )

    def test_capture_resolution_when_writing_via_slicing(self):
        """
        Scenario:  Jp2k is invoked in a write-by-slice situation and
        capture/display resolution arguments are supplied.

        Expected result:  The resolution boxes are verified.
        """
        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4

        shutil.copyfile(self.jp2file, self.temp_jp2_filename)

        j = glymur.Jp2k(
            self.temp_jp2_filename,
            capture_resolution=[vresc, hresc],
            display_resolution=[vresd, hresd],
        )

        expected = self.jp2_data
        j[:] = expected

        actual = j[:]

        np.testing.assert_array_equal(actual, expected)

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resc')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresc)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresc)

        self.assertEqual(j.box[2].box[2].box[1].box_id, 'resd')
        self.assertEqual(j.box[2].box[2].box[1].vertical_resolution, vresd)
        self.assertEqual(j.box[2].box[2].box[1].horizontal_resolution, hresd)

    def test_capture_resolution_supplied_but_not_display(self):
        """
        Scenario:  Writing a JP2 is intended, but only a capture resolution
        box is specified, and not a display resolution box.

        Expected Result:  No errors, the boxes are validated.
        """
        vresc, hresc = 0.1, 0.2

        j = glymur.Jp2k(
            self.temp_jp2_filename, data=self.jp2_data,
            capture_resolution=[vresc, hresc],
        )

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resc')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresc)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresc)

        # there's just one child box
        self.assertEqual(len(j.box[2].box[2].box), 1)

    def test_display_resolution_supplied_but_not_capture(self):
        """
        Scenario:  Writing a JP2 is intended, but only a capture resolution
        box is specified, and not a display resolution box.

        Expected Result:  No errors, the boxes are validated.
        """
        vresd, hresd = 0.3, 0.4

        j = glymur.Jp2k(
            self.temp_jp2_filename, data=self.jp2_data,
            display_resolution=[vresd, hresd],
        )

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resd')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresd)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresd)

        # there's just one child box
        self.assertEqual(len(j.box[2].box[2].box), 1)

    def test_no_jp2c_box_in_outermost_jp2_list(self):
        """
        SCENARIO:  A JP2 file is encountered without a JP2C box in the outer-
        most list of boxes.

        EXPECTED RESULT:  RuntimeError
        """
        j = glymur.Jp2k(self.jp2file)

        # Remove the last box, which is a codestream.
        boxes = j.box[:-1]

        with open(self.temp_jp2_filename, mode="wb") as tfile:
            with self.assertRaises(RuntimeError):
                j.wrap(tfile.name, boxes=boxes)

    def test_numres(self):
        """
        Scenario:  Specify numres parameter as 6.

        Expected Result:  The numres parameter will be one less.  That's ok
        because the resolutions are stored [0 .. numres-1].  That last image
        will be 1/32nd of the width/height as the original image.
        """
        expected_numres = 6
        j = glymur.Jp2k(
            self.temp_jp2_filename, data=self.jp2_data, numres=expected_numres
        )

        c = j.get_codestream()
        actual = c.segment[2].num_res
        self.assertEqual(actual, expected_numres - 1)

        # retrieve that last thumbnail
        d = j[::32, ::32]
        self.assertEqual(d.shape, (46, 81, 3))

    def test_numres_is_none(self):
        """
        Scenario:  Specify numres parameter as None

        Expected Result:  The number of decomposition levels should be five.
        It's zero-based, so the returned value is 6.
        """
        expected_numres = 6
        j = glymur.Jp2k(
            self.temp_jp2_filename, data=self.jp2_data, numres=None
        )

        c = j.get_codestream()
        actual = c.segment[2].num_res
        self.assertEqual(actual, expected_numres - 1)

    def test_null_data(self):
        """
        SCENARIO:  An image with a dimension with length 0 is provided.

        EXPECTED RESULT:  RuntimeError
        """
        with self.assertRaises(InvalidJp2kError):
            Jp2k(
                self.temp_jp2_filename,
                data=np.zeros((0, 256), dtype=np.uint8)
            )

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_psnr_zero_value_not_last(self):
        """
        SCENARIO:  The PSNR keyword argument has a zero value, but it is not
        the last value.

        EXPECTED RESULT:  RuntimeError
        """
        kwargs = {
            'data': fixtures.skimage.data.camera(),
            'psnr': [0, 35, 40, 30],
        }
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_jp2_filename, **kwargs)

    @unittest.skipIf(glymur.version.openjpeg_version < '2.5.0',
                     "Requires as least v2.5.0")
    def test_tlm_yes(self):
        """
        SCENARIO:  Use the tlm keyword.

        EXPECTED RESULT:  A TLM segment is detected.
        """
        j = Jp2k(self.temp_jp2_filename, data=self.jp2_data, tlm=True)

        codestream = j.get_codestream(header_only=False)

        at_least_one_tlm_segment = any(
            isinstance(seg, glymur.codestream.TLMsegment)
            for seg in codestream.segment
        )
        self.assertTrue(at_least_one_tlm_segment)

    def test_tlm_no(self):
        """
        SCENARIO:  Use the tlm keyword set to False

        EXPECTED RESULT:  A TLM segment not detected.
        """
        j = Jp2k(self.temp_jp2_filename, data=self.jp2_data, tlm=False)

        codestream = j.get_codestream(header_only=False)

        at_least_one_tlm_segment = any(
            isinstance(seg, glymur.codestream.TLMsegment)
            for seg in codestream.segment
        )
        self.assertFalse(at_least_one_tlm_segment)

    def test_tlm_none(self):
        """
        SCENARIO:  Use the tlm keyword set to None.  This was the default
        position in 0.12.1.

        EXPECTED RESULT:  A TLM segment not detected.
        """
        j = Jp2k(self.temp_jp2_filename, data=self.jp2_data, tlm=None)

        codestream = j.get_codestream(header_only=False)

        at_least_one_tlm_segment = any(
            isinstance(seg, glymur.codestream.TLMsegment)
            for seg in codestream.segment
        )
        self.assertFalse(at_least_one_tlm_segment)

    @unittest.skipIf(
        glymur.version.openjpeg_version < '2.4.0', "Requires as least v2.4.0"
    )
    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_plt_yes(self):
        """
        SCENARIO:  Use the plt keyword.

        EXPECTED RESULT:  Plt segment is detected.
        """
        j = Jp2k(self.temp_jp2_filename, data=self.j2k_data, plt=True)

        codestream = j.get_codestream(header_only=False)

        lst = [seg for seg in codestream.segment if seg.marker_id == 'PLT']
        self.assertEqual(len(lst), 1)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_plt_no(self):
        """
        SCENARIO:  Use the plt keyword set to false.

        EXPECTED RESULT:  Plt segment is not detected.
        """
        j = Jp2k(self.temp_jp2_filename, data=self.j2k_data, plt=False)
        codestream = j.get_codestream(header_only=False)

        at_least_one_plt = any(
            isinstance(seg, glymur.codestream.PLTsegment)
            for seg in codestream.segment
        )
        self.assertFalse(at_least_one_plt)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_plt_none(self):
        """
        SCENARIO:  Use the plt keyword set to None.

        EXPECTED RESULT:  Plt segment is not detected.
        """
        j = Jp2k(self.temp_jp2_filename, data=self.j2k_data, plt=None)
        codestream = j.get_codestream(header_only=False)

        at_least_one_plt = any(
            isinstance(seg, glymur.codestream.PLTsegment)
            for seg in codestream.segment
        )
        self.assertFalse(at_least_one_plt)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_psnr_non_zero_non_monotonically_decreasing(self):
        """
        SCENARIO:  The PSNR keyword argument is non-monotonically increasing
        and does not contain zero.

        EXPECTED RESULT:  RuntimeError
        """
        kwargs = {
            'data': fixtures.skimage.data.camera(),
            'psnr': [30, 35, 40, 30],
        }
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_jp2_filename, **kwargs)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_psnr(self):
        """
        SCENARIO:  Four peak signal-to-noise ratio values are supplied, the
        last is zero.

        EXPECTED RESULT:  Four quality layers, the first should be lossless.
        """
        kwargs = {
            'data': fixtures.skimage.data.camera(),
            'psnr': [30, 35, 40, 0],
        }
        j = Jp2k(self.temp_jp2_filename, **kwargs)

        d = {}
        for layer in range(4):
            j.layer = layer
            d[layer] = j[:]

        with warnings.catch_warnings():
            # MSE is zero for that first image, resulting in a divide-by-zero
            # warning
            warnings.simplefilter('ignore')
            psnr = [
                fixtures.skimage.metrics.peak_signal_noise_ratio(
                    fixtures.skimage.data.camera(), d[j]
                )
                for j in range(4)
            ]

        # That first image should be lossless.
        self.assertTrue(np.isinf(psnr[0]))

        # None of the subsequent images should have inf PSNR.
        self.assertTrue(not np.any(np.isinf(psnr[1:])))

        # PSNR should increase for the remaining images.
        self.assertTrue(np.all(np.diff(psnr[1:])) > 0)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_psnr_from_doctest(self):
        """
        SCENARIO:  Four peak signal-to-noise ratio values are supplied, the
        last is zero.

        EXPECTED RESULT:  Four quality layers, the first should be lossless.
        """
        kwargs = {
            'data': fixtures.skimage.data.camera(),
            'psnr': [30, 40, 50, 0],
        }
        j = Jp2k(self.temp_jp2_filename, **kwargs)

        d = {}
        for layer in range(4):
            j.layer = layer
            d[layer] = j[:]

        with warnings.catch_warnings():
            # MSE is zero for that first image, resulting in a divide-by-zero
            # warning
            warnings.simplefilter('ignore')
            psnr = [
                fixtures.skimage.metrics.peak_signal_noise_ratio(
                    fixtures.skimage.data.camera(), d[j]
                )
                for j in range(4)
            ]

        # That first image should be lossless.
        self.assertTrue(np.isinf(psnr[0]))

        # None of the subsequent images should have inf PSNR.
        self.assertTrue(not np.any(np.isinf(psnr[1:])))

        # PSNR should increase for the remaining images.
        self.assertTrue(np.all(np.diff(psnr[1:])) > 0)

    def test_NR_ENC_Bretagne1_ppm_2_encode(self):
        """
        SCENARIO:  Three peak signal-to-noise ratio values, two resolutions are
        supplied.

        EXPECTED RESULT:  Three quality layers, two resolutions.
        """
        kwargs = {
            'data': self.jp2_data,
            'psnr': [30, 35, 40],
            'numres': 2,
        }
        j = Jp2k(self.temp_j2k_filename, **kwargs)
        codestream = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res + 1, 2)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size, np.array(((32768, 32768)))
        )

    def test_NR_ENC_Bretagne1_ppm_1_encode(self):
        """
        SCENARIO:  Create a JP2 image with three compression ratios.

        EXPECTED RESULT:  There are three layers.
        """
        data = self.jp2_data
        # Should be written with 3 layers.
        j = Jp2k(self.temp_j2k_filename, data=data, cratios=(200, 100, 50))
        c = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].mct, 1)  # mct
        self.assertEqual(c.segment[2].num_res + 1, 6)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self._verify_codeblock_style(
            c.segment[2].cstyle, [False, False, False, False, False, False]
        )
        self.assertEqual(c.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            c.segment[2].precinct_size, np.array((32768, 32768))
        )

    def test_NR_ENC_Bretagne1_ppm_3_encode(self):
        """
        SCENARIO:  Three peak signal to noise rations are provided, along with
        specific code block sizes and precinct sizes.

        EXPECTED RESULT:  Three quality layers and the specified code block
        size are present.  The precinct sizes validate.
        """
        j = Jp2k(
            self.temp_j2k_filename,
            data=self.jp2_data,
            psnr=[30, 35, 40],
            cbsize=(16, 16), psizes=[(64, 64)]
        )

        codestream = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (16, 16))  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size,
            np.array(((2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64)))
        )

    def test_NR_ENC_Bretagne2_ppm_4_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        j = Jp2k(
            self.temp_j2k_filename,
            data=self.jp2_data,
            psizes=[(128, 128)] * 3,
            cratios=(100, 20, 2),
            tilesize=(480, 640),
            cbsize=(32, 32)
        )

        # Should be three layers.
        codestream = j.get_codestream()

        # RSIZ
        self.assertEqual(codestream.segment[1].xtsiz, 640)
        self.assertEqual(codestream.segment[1].ytsiz, 480)

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order,
                         glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(
            tuple(codestream.segment[2].code_block_size),
            (32, 32)
        )  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(
            codestream.segment[2].xform,
            glymur.core.WAVELET_XFORM_5X3_REVERSIBLE
        )
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size,
            np.array((
                (16, 16), (32, 32), (64, 64), (128, 128), (128, 128),
                (128, 128)
            ))
        )

    def test_NR_ENC_Bretagne2_ppm_5_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data,
                 tilesize=(127, 127), prog="PCRL")

        codestream = j.get_codestream()

        # RSIZ
        self.assertEqual(codestream.segment[1].xtsiz, 127)
        self.assertEqual(codestream.segment[1].ytsiz, 127)

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order,
                         glymur.core.PCRL)
        self.assertEqual(codestream.segment[2].layers, 1)
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size,
            np.array(((32768, 32768)))
        )

    def test_sop_explicitly_true(self):
        """
        Scenario:   Specify sop=True

        Expected Result:  There are 17 SOP packets in the codestream.
        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, sop=True)

        codestream = j.get_codestream(header_only=False)

        lst = [seg for seg in codestream.segment if seg.marker_id == 'SOP']
        self.assertEqual(len(lst), 18)

    def test_sop_explicitly_false(self):
        """
        Scenario:   Specify sop=False

        Expected Result:  There are no SOP packets in the codestream.
        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, sop=False)

        codestream = j.get_codestream(header_only=False)

        b = any(seg for seg in codestream.segment if seg.marker_id == 'SOP')
        self.assertFalse(b)

    def test_sop_explicitly_none(self):
        """
        Scenario:   Specify sop=None

        Expected Result:  There are no SOP packets in the codestream.  This is
        the old behavior.
        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, sop=None)

        codestream = j.get_codestream(header_only=False)

        b = any(seg for seg in codestream.segment if seg.marker_id == 'SOP')
        self.assertFalse(b)

    def test_NR_ENC_Bretagne2_ppm_6_encode(self):
        """
        Scenario:   Specify subsampling and writing SOP markers before each
        packet.

        Expected Result:  Subsampling is verified.  Usage of SOP is verified.
        The original file tested in the openjpeg test suite was
        input/nonregression/Bretagne2.ppm
        """
        j = Jp2k(
            self.temp_j2k_filename,
            data=self.jp2_data, subsam=(2, 2), sop=True
        )

        codestream = j.get_codestream(header_only=False)

        # RSIZ
        self.assertEqual(codestream.segment[1].xrsiz, (2, 2, 2))
        self.assertEqual(codestream.segment[1].yrsiz, (2, 2, 2))

        # COD: Coding style default
        self.assertTrue(codestream.segment[2].scod & 2)  # sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order,
                         glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size,
            np.array(((32768, 32768)))
        )

        # 18 SOP segments.
        nsops = [x.nsop for x in codestream.segment
                 if x.marker_id == 'SOP']
        self.assertEqual(nsops, list(range(18)))

    def test_eph(self):
        """
        Scenario:  eph is True

        Expected Result:  EPH markers expected in the codestream
        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, eph=True)

        codestream = j.get_codestream(header_only=False)

        self.assertTrue(codestream.segment[2].scod & 4)  # eph

        # 18 EPH segments.
        ephs = [x for x in codestream.segment if x.marker_id == 'EPH']
        self.assertEqual(len(ephs), 18)

    def test_no_eph(self):
        """
        Scenario:  eph is False

        Expected Result:  No EPH markers expected in the codestream
        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, eph=False)

        codestream = j.get_codestream(header_only=False)

        self.assertFalse(codestream.segment[2].scod & 4)  # eph

        # No EPH segments.
        ephs = [x for x in codestream.segment if x.marker_id == 'EPH']
        self.assertEqual(len(ephs), 0)

    def test_eph_is_none(self):
        """
        Scenario:  eph is None.  This is a test for backwards compatibility,
        as eph is documented as a bool-only parameter as of version > 0.2.12.

        Expected Result:  No EPH markers expected in the codestream
        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, eph=None)

        codestream = j.get_codestream(header_only=False)

        self.assertFalse(codestream.segment[2].scod & 4)  # eph

        # No EPH segments.
        ephs = [x for x in codestream.segment if x.marker_id == 'EPH']
        self.assertEqual(len(ephs), 0)

    def test_NR_ENC_Bretagne2_ppm_7_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        j = Jp2k(
            self.temp_j2k_filename, data=self.jp2_data, modesw=38, eph=True
        )

        codestream = j.get_codestream(header_only=False)

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertTrue(codestream.segment[2].scod & 4)  # eph
        self.assertEqual(codestream.segment[2].prog_order,
                         glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, True, True, False, False, True]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size,
            np.array(((32768, 32768)))
        )

        # 18 EPH segments.
        ephs = [x for x in codestream.segment if x.marker_id == 'EPH']
        self.assertEqual(len(ephs), 18)

    def test_modeswitch_specified(self):
        """
        Scenario:  specify a modeswitch of 38 (RESTART + RESET + SEGMARK)

        Expected Result: cstyle of 38 is verified.
        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, modesw=38)

        codestream = j.get_codestream(header_only=False)

        self.assertEqual(codestream.segment[2].code_block_size, (64, 64))
        self.assertEqual(codestream.segment[2].cstyle, 38)

    def test_modeswitch_default(self):
        """
        Scenario:  specify a modeswitch of 0 (default)

        Expected Result:  0
        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, modesw=0)

        codestream = j.get_codestream(header_only=False)
        self.assertEqual(codestream.segment[2].cstyle, 0)

    def test_modeswitch_none(self):
        """
        Scenario:  none was the old default for modeswitch

        Expected Result:  0
        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, modesw=None)

        codestream = j.get_codestream(header_only=False)
        self.assertEqual(codestream.segment[2].cstyle, 0)

    def test_NR_ENC_Bretagne2_ppm_8_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm
        """
        j = Jp2k(self.temp_j2k_filename,
                 data=self.jp2_data, grid_offset=[300, 150], cratios=(800,))

        codestream = j.get_codestream(header_only=False)

        # RSIZ
        self.assertEqual(codestream.segment[1].xosiz, 150)
        self.assertEqual(codestream.segment[1].yosiz, 300)

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order,
                         glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size,
            np.array(((32768, 32768)))
        )

    def test_NR_ENC_Cevennes1_bmp_9_encode(self):
        """
        Original file tested was

            input/nonregression/Cevennes1.bmp

        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, cratios=(800,))

        codestream = j.get_codestream(header_only=False)

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order,
                         glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size,
            np.array(((32768, 32768)))
        )

    def test_NR_ENC_Cevennes2_ppm_10_encode(self):
        """
        Original file tested was

            input/nonregression/Cevennes2.ppm

        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, cratios=(50,))

        codestream = j.get_codestream(header_only=False)

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order,
                         glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size,
            np.array(((32768, 32768)))
        )

    def test_NR_ENC_Rome_bmp_11_encode(self):
        """
        Original file tested was

            input/nonregression/Rome.bmp

        """
        jp2 = Jp2k(
            self.temp_jp2_filename,
            data=self.jp2_data, psnr=[30, 35, 50], prog='LRCP', numres=3
        )

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 1456)
        self.assertEqual(jp2.box[2].box[0].width, 2592)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)
        self.assertIsNone(jp2.box[2].box[1].icc_profile)
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)

        codestream = jp2.box[3].codestream

        kwargs = {
            'rsiz': 0,
            'xysiz': (2592, 1456),
            'xyosiz': (0, 0),
            'xytsiz': (2592, 1456),
            'xytosiz': (0, 0),
            'bitdepth': (8, 8, 8),
            'signed': (False, False, False),
            'xyrsiz': [(1, 1, 1), (1, 1, 1)]
        }
        self._verify_size_segment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order,
                         glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 2)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size,
            np.array(((32768, 32768)))
        )

    def test_NR_ENC_random_issue_0005_tif_12_encode(self):
        """
        Original file tested was

            input/nonregression/random-issue-0005.tif
        """
        data = self.jp2_data[:1024, :1024, 0].astype(np.uint16)
        j = Jp2k(self.temp_j2k_filename, data=data)

        codestream = j.get_codestream(header_only=False)

        kwargs = {
            'rsiz': 0,
            'xysiz': (1024, 1024),
            'xyosiz': (0, 0),
            'xytsiz': (1024, 1024),
            'xytosiz': (0, 0),
            'bitdepth': (16,),
            'signed': (False,),
            'xyrsiz': [(1,), (1,)]
        }
        self._verify_size_segment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order,
                         glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
        self.assertEqual(codestream.segment[2].mct, 0)
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self._verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        np.testing.assert_array_equal(
            codestream.segment[2].precinct_size, np.array((32768, 32768))
        )

    def test_NR_ENC_issue141_rawl_23_encode(self):
        """
        Test irreversible option

        Original file tested was

            input/nonregression/issue141.rawl

        """
        j = Jp2k(self.temp_j2k_filename, data=self.jp2_data, irreversible=True)

        codestream = j.get_codestream()
        self.assertEqual(
            codestream.segment[2].xform,
            glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE
        )

    def test_cinema2K_with_others(self):
        """
        Can't specify cinema2k with any other options.

        Original test file was
        input/nonregression/X_5_2K_24_235_CBR_STEM24_000.tif
        """
        data = np.zeros((857, 2048, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename, data=data,
                 cinema2k=48, cratios=(200, 100, 50))

    def test_cinema4K_with_others(self):
        """
        Can't specify cinema4k with any other options.

        Original test file was input/nonregression/ElephantDream_4K.tif
        """
        data = np.zeros((4096, 2160, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename, data=data,
                 cinema4k=True, cratios=(200, 100, 50))

    def test_cblk_size_precinct_size(self):
        """
        code block sizes should never exceed half that of precinct size.
        """
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename, data=self.j2k_data,
                 cbsize=(64, 64), psizes=[(64, 64)])

    def test_cblk_size_not_power_of_two(self):
        """
        code block sizes should be powers of two.
        """
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename, data=self.j2k_data, cbsize=(13, 12))

    def test_precinct_size_not_p2(self):
        """
        precinct sizes should be powers of two.
        """
        with self.assertRaises(RuntimeError):
            Jp2k(
                self.temp_j2k_filename, data=self.j2k_data, psizes=[(173, 173)]
            )

    def test_code_block_dimensions(self):
        """
        don't allow extreme codeblock sizes
        """
        # opj_compress doesn't allow the dimensions of a codeblock
        # to be too small or too big, so neither will we.
        data = self.j2k_data

        # opj_compress doesn't allow code block area to exceed 4096.
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename, data=data, cbsize=(256, 256))

        # opj_compress doesn't allow either dimension to be less than 4.
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename, data=data, cbsize=(2048, 2))

        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename, data=data, cbsize=(2, 2048))

    def test_psnr_with_cratios(self):
        """
        Using psnr with cratios options is not allowed.
        """
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename, data=self.j2k_data, psnr=[30, 35, 40],
                 cratios=(2, 3, 4))

    def test_irreversible(self):
        """
        Verify that the Irreversible option works
        """
        expdata = self.j2k_data
        j = Jp2k(
            self.temp_j2k_filename, data=expdata, irreversible=True, numres=5
        )

        codestream = j.get_codestream()
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

        actdata = j[:]

        diff = actdata.astype(np.double) - expdata.astype(np.double)
        mse = np.mean(diff**2)

        self.assertTrue(mse < 0.28)

    def test_shape_greyscale_jp2(self):
        """verify shape attribute for greyscale JP2 file
        """
        jp2 = Jp2k(self.single_channel_jp2)
        self.assertEqual(jp2.shape, (800, 480))
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_shape_single_channel_j2k(self):
        """verify shape attribute for single channel J2K file
        """
        j2k = Jp2k(self.single_channel_j2k)
        self.assertEqual(j2k.shape, (800, 480))

    def test_precinct_size_too_small(self):
        """
        SCENARIO:  The first precinct size is less than 2x that of the code
        block size.

        EXPECTED RESULT:  InvalidJp2kError
        """
        data = np.zeros((640, 480), dtype=np.uint8)
        with self.assertRaises(InvalidJp2kError):
            Jp2k(
                self.temp_j2k_filename,
                data=data, cbsize=(16, 16), psizes=[(16, 16)]
            )

    def test_precinct_size_not_power_of_two(self):
        """
        SCENARIO:  A precinct size is specified that is not a power of 2.

        EXPECTED RESULT:  InvalidJp2kError
        """
        data = np.zeros((640, 480), dtype=np.uint8)
        with self.assertRaises(InvalidJp2kError):
            Jp2k(
                self.temp_j2k_filename, data=data, cbsize=(16, 16),
                psizes=[(48, 48)]
            )

    def test_unsupported_int32(self):
        """Should raise a runtime error if trying to write int32"""
        data = np.zeros((128, 128), dtype=np.int32)
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename, data=data)

    def test_unsupported_uint32(self):
        """Should raise a runtime error if trying to write uint32"""
        data = np.zeros((128, 128), dtype=np.uint32)
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename, data=data)

    def test_write_with_version_too_early(self):
        """Should raise a runtime error if trying to write with version 1.3"""
        data = np.zeros((128, 128), dtype=np.uint8)
        versions = [
            "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "2.0.0",
            "2.1.0", "2.2.0"
        ]
        for version in versions:
            with patch('glymur.version.openjpeg_version', new=version):
                with self.assertRaises(RuntimeError):
                    Jp2k(self.temp_j2k_filename, data=data)

    def test_cblkh_different_than_width(self):
        """Verify that we can set a code block size where height does not equal
        width.
        """
        data = np.zeros((128, 128), dtype=np.uint8)
        # The code block dimensions are given as rows x columns.
        j = Jp2k(self.temp_j2k_filename, data=data, cbsize=(16, 32))
        codestream = j.get_codestream()

        # Code block size is reported as XY in the codestream.
        self.assertEqual(codestream.segment[2].code_block_size, (16, 32))

    def test_too_many_dimensions(self):
        """OpenJP2 only allows 2D or 3D images."""
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename,
                 data=np.zeros((128, 128, 2, 2), dtype=np.uint8))

    def test_2d_rgb(self):
        """RGB must have at least 3 components."""
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_jp2_filename,
                 data=np.zeros((128, 128, 2), dtype=np.uint8),
                 colorspace='rgb')

    def test_colorspace_with_j2k(self):
        """Specifying a colorspace with J2K does not make sense"""
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_j2k_filename,
                 data=np.zeros((128, 128, 3), dtype=np.uint8),
                 colorspace='rgb')

    def test_specify_rgb(self):
        """specify RGB explicitly"""
        j = Jp2k(self.temp_jp2_filename,
                 data=np.zeros((128, 128, 3), dtype=np.uint8),
                 colorspace='rgb')
        self.assertEqual(j.box[2].box[1].colorspace, glymur.core.SRGB)

    def test_colorspace_is_explicitly_none_but_3D_image(self):
        """
        Scenario:  specify None for the colorspace, but give a 3D image data
        as input

        Expected result:  the colorspace is SRGB
        """
        j = Jp2k(
            self.temp_jp2_filename,
            data=np.zeros((128, 128, 3), dtype=np.uint8),
            colorspace=None
        )
        self.assertEqual(j.box[2].box[1].colorspace, glymur.core.SRGB)

    def test_specify_gray(self):
        """test gray explicitly specified (that's GRAY, not GREY)"""
        data = np.zeros((128, 128), dtype=np.uint8)
        j = Jp2k(self.temp_jp2_filename, data=data, colorspace='gray')
        self.assertEqual(j.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_specify_grey(self):
        """test grey explicitly specified"""
        data = np.zeros((128, 128), dtype=np.uint8)
        j = Jp2k(self.temp_jp2_filename, data=data, colorspace='grey')
        self.assertEqual(j.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_grey_with_two_extra_comps(self):
        """should be able to write gray + two extra components"""
        data = np.zeros((128, 128, 3), dtype=np.uint8)
        j = Jp2k(self.temp_jp2_filename, data=data, colorspace='gray')
        self.assertEqual(j.box[2].box[0].height, 128)
        self.assertEqual(j.box[2].box[0].width, 128)
        self.assertEqual(j.box[2].box[0].num_components, 3)
        self.assertEqual(j.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_specify_ycc(self):
        """Should reject YCC"""
        data = np.zeros((128, 128, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_jp2_filename, data=data, colorspace='ycc')

    def test_write_with_jp2_in_caps(self):
        """should be able to write with JP2 suffix."""
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]

        filename = str(self.temp_jp2_filename).replace('.jp2', '.JP2')

        ofile = Jp2k(filename, data=expdata)
        actdata = ofile[:]
        np.testing.assert_array_equal(actdata, expdata)

    def test_write_srgb_specifying_mct_as_none(self):
        """
        Scenario:  Write RGB data, explicitly settimg mct to None, which is old
        behavior.

        Expected Result:  The codestream should record mct as being set.
        """
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]
        ofile = Jp2k(self.temp_jp2_filename, data=expdata, mct=None)
        actdata = ofile[:]
        np.testing.assert_array_equal(actdata, expdata)

        codestream = ofile.get_codestream()
        self.assertEqual(codestream.segment[2].mct, 1)

    def test_write_srgb_without_mct(self):
        """
        Scenario:  Write RGB data, set mct to False.

        Expected Result:  The codestream records mct as not being set.
        """
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]
        ofile = Jp2k(self.temp_jp2_filename, data=expdata, mct=False)
        actdata = ofile[:]
        np.testing.assert_array_equal(actdata, expdata)

        codestream = ofile.get_codestream()
        self.assertEqual(codestream.segment[2].mct, 0)  # no mct

    def test_write_grayscale_with_mct(self):
        """
        Scenario:  Explicitly specify mct for a grayscale image.  The MCT does
        not make sense there.

        Expected Result:  RuntimeError
        """
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_jp2_filename, data=expdata[:, :, 0], mct=True)

    def test_write_grayscale_with_mct_set_to_none(self):
        """
        Scenario:  Explicitly specify mct as None for a grayscale image.

        Expected Result:  The MCT is not used.
        """
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:][:, :, 2]

        j = Jp2k(self.temp_jp2_filename, data=expdata, mct=None)

        actdata = j[:]
        np.testing.assert_array_equal(actdata, expdata)

        codestream = j.get_codestream()
        self.assertEqual(codestream.segment[2].mct, 0)  # no mct

    def test_write_grayscale_with_mct_set_to_false(self):
        """
        Scenario:  Explicitly specify mct as False for a grayscale image.

        Expected Result:  The MCT is not used.  That's the default for
        grayscale anyway.
        """
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:][:, :, 2]

        j = Jp2k(self.temp_jp2_filename, data=expdata, mct=False)

        actdata = j[:]
        np.testing.assert_array_equal(actdata, expdata)

        codestream = j.get_codestream()
        self.assertEqual(codestream.segment[2].mct, 0)  # no mct

    def test_write_cprl(self):
        """Must be able to write a CPRL progression order file"""
        # Issue 17
        j = Jp2k(self.jp2file)
        expdata = j[::2, ::2]
        ofile = Jp2k(self.temp_jp2_filename, data=expdata, prog='CPRL')
        actdata = ofile[:]
        np.testing.assert_array_equal(actdata, expdata)

        codestream = ofile.get_codestream()
        self.assertEqual(codestream.segment[2].prog_order, glymur.core.CPRL)

    def test_bad_area_parameter(self):
        """Should error out appropriately if given a bad area parameter."""
        j = Jp2k(self.jp2file)
        error = glymur.lib.openjp2.OpenJPEGLibraryError
        with self.assertRaises(ValueError):
            # Start corner must be >= 0
            j[-1:1, -1:1]
        with self.assertRaises(ValueError):
            # End corner must be > 0
            j[10:0, 10:0]
        with self.assertRaises(error):
            # End corner must be >= start corner
            j[10:8, 10:8]

    def test_unrecognized_jp2_clrspace(self):
        """We only allow RGB and GRAYSCALE.  Should error out with others"""
        data = np.zeros((128, 128, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_jp2_filename, data=data, colorspace='cmyk')

    def test_asoc_label_box(self):
        """Test asoc and label box"""
        # Construct a fake file with an asoc and a label box, as
        # OpenJPEG doesn't have such a file.
        data = Jp2k(self.jp2file)[::2, ::2]
        file1 = self.test_dir_path / 'file1.jp2'
        Jp2k(file1, data=data)

        with open(file1, mode='rb') as tfile:

            file2 = self.test_dir_path / 'file2.jp2'
            with open(file2, mode='wb') as tfile2:

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

    def test_grey_with_extra_component(self):
        """version 2.0 cannot write gray + extra"""
        data = np.zeros((128, 128, 2), dtype=np.uint8)
        j = Jp2k(self.temp_jp2_filename, data=data)
        self.assertEqual(j.box[2].box[0].height, 128)
        self.assertEqual(j.box[2].box[0].width, 128)
        self.assertEqual(j.box[2].box[0].num_components, 2)
        self.assertEqual(j.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_rgb_with_extra_component(self):
        """v2.0+ should be able to write extra components"""
        data = np.zeros((128, 128, 4), dtype=np.uint8)
        j = Jp2k(self.temp_jp2_filename, data=data)
        self.assertEqual(j.box[2].box[0].height, 128)
        self.assertEqual(j.box[2].box[0].width, 128)
        self.assertEqual(j.box[2].box[0].num_components, 4)
        self.assertEqual(j.box[2].box[1].colorspace, glymur.core.SRGB)

    def test_openjpeg_library_error(self):
        """
        SCENARIO:  A zero subsampling factor should produce as error by the
        library.

        EXPECTED RESULT:  OpenJPEGLibraryError
        """
        # This will confirm that the error callback mechanism is working.
        with open(self.jp2file, 'rb') as fptr:
            data = fptr.read()
            with open(self.temp_jp2_filename, mode='wb') as tfile:
                # Codestream starts at byte 3223. SIZ marker at 3233.
                # COD marker at 3282.  Subsampling at 3276.
                # Codestream starts at byte 77. SIZ marker at 10.
                # COD marker at 59.  Subsampling at 53.
                offset = 77
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
                    error = glymur.lib.openjp2.OpenJPEGLibraryError
                    with self.assertRaises(error):
                        j[::2, ::2]

    def test_astronaut(self):
        """
        SCENARIO:  construct a j2k file by tiling an image in a 2x2 grid.

        EXPECTED RESULT:  the written image validates
        """
        j2k_data = fixtures.skimage.data.astronaut()
        data = [
            j2k_data[:256, :256, :],
            j2k_data[:256, 256:512, :],
            j2k_data[256:512, :256, :],
            j2k_data[256:512, 256:512, :],
        ]

        shape = j2k_data.shape
        tilesize = 256, 256

        j = Jp2k(self.temp_j2k_filename, shape=shape, tilesize=tilesize)
        for idx, tw in enumerate(j.get_tilewriters()):
            tw[:] = data[idx]

        new_j = Jp2k(self.temp_j2k_filename)
        actual = new_j[:]
        expected = j2k_data
        np.testing.assert_array_equal(actual, expected)

    def test_smoke(self):
        """
        SCENARIO:  construct a j2k file by repeating a 3D image in a 2x2 grid.

        EXPECTED RESULT:  the written image matches the 2x2 grid
        """
        j2k_data = fixtures.skimage.data.astronaut()

        shape = (
            j2k_data.shape[0] * 2, j2k_data.shape[1] * 2, j2k_data.shape[2]
        )
        tilesize = (j2k_data.shape[0], j2k_data.shape[1])

        j = Jp2k(self.temp_j2k_filename, shape=shape, tilesize=tilesize)
        for tw in j.get_tilewriters():
            tw[:] = j2k_data

        new_j = Jp2k(self.temp_j2k_filename)
        actual = new_j[:]
        expected = np.tile(j2k_data, (2, 2, 1))
        np.testing.assert_array_equal(actual, expected)

    def test_moon(self):
        """
        SCENARIO:  construct a jp2 file by repeating a 2D image in a 3x2 grid.

        EXPECTED RESULT:  the written image matches the 3x2 grid
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 3, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = Jp2k(self.temp_jp2_filename, shape=shape, tilesize=tilesize)
        for tw in j.get_tilewriters():
            tw[:] = jp2_data

        new_j = Jp2k(self.temp_jp2_filename)
        actual = new_j[:]
        expected = np.tile(jp2_data, (3, 2))
        np.testing.assert_array_equal(actual, expected)

    def test_tile_slice_has_non_none_elements(self):
        """
        SCENARIO:  construct a jp2 file by repeating a 2D image in a 2x2 grid,
        but the tile writer does not receive a degenerate slice object.

        EXPECTED RESULT:  RuntimeError
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 2, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = Jp2k(self.temp_jp2_filename, shape=shape, tilesize=tilesize)
        with self.assertRaises(RuntimeError):
            for tw in j.get_tilewriters():
                tw[:256, :256] = jp2_data

    def test_tile_slice_is_ellipsis(self):
        """
        SCENARIO:  construct a jp2 file by repeating a 2D image in a 2x2 grid,
        but the tile writer does not receive a degenerate slice object.

        EXPECTED RESULT:  RuntimeError
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 2, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = Jp2k(self.temp_jp2_filename, shape=shape, tilesize=tilesize)
        with self.assertRaises(RuntimeError):
            for tw in j.get_tilewriters():
                tw[...] = jp2_data

    def test_too_much_data_for_slice(self):
        """
        SCENARIO:  construct a jp2 file by repeating a 2D image in a 2x2 grid,
        but the tile writer does not receive a degenerate slice object.

        EXPECTED RESULT:  RuntimeError
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 2, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = Jp2k(self.temp_jp2_filename, shape=shape, tilesize=tilesize)
        with self.assertRaises(glymur.lib.openjp2.OpenJPEGLibraryError):
            for tw in j.get_tilewriters():
                tw[:] = np.tile(jp2_data, (2, 2))

    def test_write_with_different_compression_ratios(self):
        """
        SCENARIO:  construct a jp2 file by repeating a 2D image in a 2x2 grid.

        EXPECTED RESULT:  There are three layers.
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 2, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = Jp2k(
            self.temp_jp2_filename, shape=shape, tilesize=tilesize,
            cratios=[20, 5, 1]
        )

        for tw in j.get_tilewriters():
            tw[:] = jp2_data

        codestream = j.get_codestream()
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3

    def test_capture_resolution_and_tiled_writing(self):
        """
        SCENARIO:  Use the capture_resolution keyword.

        EXPECTED RESULT:  The resolution superbox, along with a capture
        box, is inserted into the jp2 header box.
        """
        j2k_data = fixtures.skimage.data.astronaut()

        shape = (
            j2k_data.shape[0] * 2, j2k_data.shape[1] * 2, j2k_data.shape[2]
        )
        tilesize = (j2k_data.shape[0], j2k_data.shape[1])

        vresc, hresc = 0.1, 0.2

        j = glymur.Jp2k(
            self.temp_jp2_filename, shape=shape, tilesize=tilesize,
            capture_resolution=[vresc, hresc],
        )

        for tw in j.get_tilewriters():
            tw[:] = j2k_data

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resc')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresc)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresc)

    @unittest.skipIf(
        glymur.version.openjpeg_version < '2.4.0', "Requires as least v2.4.0"
    )
    def test_plt_for_tiled_writing(self):
        """
        SCENARIO:  Use the plt keyword.

        EXPECTED RESULT:  Plt segment is detected.
        """
        j2k_data = fixtures.skimage.data.astronaut()

        shape = (
            j2k_data.shape[0] * 2, j2k_data.shape[1] * 2, j2k_data.shape[2]
        )
        tilesize = (j2k_data.shape[0], j2k_data.shape[1])

        j = Jp2k(
            self.temp_j2k_filename, shape=shape, tilesize=tilesize,
            plt=True
        )
        for tw in j.get_tilewriters():
            tw[:] = j2k_data

        codestream = j.get_codestream(header_only=False)

        at_least_one_plt = any(
            isinstance(seg, glymur.codestream.PLTsegment)
            for seg in codestream.segment
        )
        self.assertTrue(at_least_one_plt)

    def test_1x1_tile(self):
        """
        SCENARIO:  Write by tiles an image that is tiled 1x1.

        EXPECTED RESULT:  RuntimeError, as this triggers an unresolved
        bug, issue586.
        """
        j2k_data = fixtures.skimage.data.astronaut()

        shape = (
            j2k_data.shape[0], j2k_data.shape[1], j2k_data.shape[2]
        )
        tilesize = (j2k_data.shape[0], j2k_data.shape[1])

        j = Jp2k(
            self.temp_j2k_filename, shape=shape, tilesize=tilesize,
        )
        with self.assertRaises(RuntimeError):
            j.get_tilewriters()

    def test_openjpeg_library_too_old_for_tile_writing(self):
        """
        SCENARIO:  Try to create a jp2 file via writing tiles, but the openjpeg
        library is too old.

        EXPECTED RESULT:  RuntimeError.
        """
        j2k_data = fixtures.skimage.data.astronaut()

        shape = (
            j2k_data.shape[0] * 2, j2k_data.shape[1] * 2, j2k_data.shape[2]
        )
        tilesize = (j2k_data.shape[0], j2k_data.shape[1])

        j = Jp2k(
            self.temp_j2k_filename, shape=shape, tilesize=tilesize,
        )
        with patch('glymur.version.openjpeg_version', new='2.2.0'):
            with self.assertRaises(RuntimeError):
                for tw in j.get_tilewriters():
                    tw[:] = j2k_data
