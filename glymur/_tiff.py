# -*- coding:  utf-8 -*-
"""
Part of glymur.
"""
# Standard library imports ...
from collections import OrderedDict
import struct
import warnings

# 3rd party library imports
import numpy as np


def tiff_header(read_buffer):
    """
    Interpret the uuid raw data as a tiff header.
    """
    # First 8 should be (73, 73, 42, 8) or (77, 77, 42, 8)
    data = struct.unpack('BB', read_buffer[0:2])
    if data[0] == 73 and data[1] == 73:
        # little endian
        endian = '<'
    elif data[0] == 77 and data[1] == 77:
        # big endian
        endian = '>'
    else:
        msg = (
            f"The byte order indication in the TIFF header "
            f"({read_buffer[0:2]}) is invalid.  It should be either "
            f"{bytes([73, 73])} or {bytes([77, 77])}."
        )
        raise RuntimeError(msg)

    _, offset = struct.unpack(endian + 'HI', read_buffer[2:8])

    # This is the 'Exif Image' portion.
    return Ifd(endian, read_buffer, offset).processed_ifd


class BadTiffTagDatatype(RuntimeError):
    """
    This exception exists soley to better communicate up the stack that the
    problem exists.
    """
    pass


class Ifd(object):
    """
    Attributes
    ----------
    read_buffer : bytes
        Raw byte stream consisting of the UUID data.
    endian : str
        Either '<' for big-endian, or '>' for little-endian.
    num_tags : int
        Number of tags in the IFD.
    raw_ifd : dictionary
        Maps tag number to "mildly-interpreted" tag value.
    processed_ifd : dictionary
        Maps tag name to "mildly-interpreted" tag value.
    """
    def __init__(self, endian, read_buffer, offset):
        self.endian = endian
        self.read_buffer = read_buffer
        self.processed_ifd = OrderedDict()

        self.num_tags, = struct.unpack(
            endian + 'H', read_buffer[offset:offset + 2]
        )

        fmt = self.endian + 'HHII' * self.num_tags
        ifd_buffer = read_buffer[offset + 2:offset + 2 + self.num_tags * 12]
        data = struct.unpack(fmt, ifd_buffer)
        self.raw_ifd = OrderedDict()
        for j, tag in enumerate(data[0::4]):
            # The offset to the tag offset/payload is the offset to the IFD
            # plus 2 bytes for the number of tags plus 12 bytes for each
            # tag entry plus 8 bytes to the offset/payload itself.
            toffp = read_buffer[offset + 10 + j * 12:offset + 10 + j * 12 + 4]
            self.raw_ifd[tag] = self.parse_tag(
                tag, data[j * 4 + 1], data[j * 4 + 2], toffp
            )

        self.post_process()

    def parse_tag(self, tag, dtype, count, offset_buf):
        """
        Interpret an Exif image tag data payload.
        """

        try:
            fmt = DATATYPE2FMT[dtype][0] * count
            payload_size = DATATYPE2FMT[dtype][1] * count
        except KeyError:
            msg = f'Invalid TIFF tag datatype ({dtype}).'
            raise BadTiffTagDatatype(msg)

        if payload_size <= 4:
            # Interpret the payload from the 4 bytes in the tag entry.
            target_buffer = offset_buf[:payload_size]
        else:
            # Interpret the payload at the offset specified by the 4 bytes in
            # the tag entry.
            offset, = struct.unpack(self.endian + 'I', offset_buf)
            target_buffer = self.read_buffer[offset:offset + payload_size]

        if dtype == 2:
            # ASCII
            payload = target_buffer.decode('utf-8').rstrip('\x00')

        else:
            payload = struct.unpack(self.endian + fmt, target_buffer)
            if dtype == 5 or dtype == 10:
                # Rational or Signed Rational.  Construct the list of values.
                rational_payload = []
                for j in range(count):
                    value = float(payload[j * 2]) / float(payload[j * 2 + 1])
                    rational_payload.append(value)
                payload = np.array(rational_payload)
            if count == 1:
                # If just a single value, then return a scalar instead of a
                # tuple.
                payload = payload[0]
            else:
                payload = np.array(payload, dtype=TIFFTYPE2NP[dtype])

        return payload

    def post_process(self):
        """
        Map the tag name instead of tag number to the tag value.
        """
        for tag, value in self.raw_ifd.items():
            try:
                tag_name = TAGNUM2NAME[tag]
            except KeyError:
                # Ok, we don't recognize this tag.  Just use the numeric id.
                msg = f'Unrecognized UUID box TIFF tag ({tag}).'
                warnings.warn(msg, UserWarning)
                tag_name = tag

            if tag_name == 'ExifTag':
                # There's an Exif IFD at the offset specified here.
                ifd = Ifd(self.endian, self.read_buffer, value)
                self.processed_ifd[tag_name] = ifd.processed_ifd
            else:
                # just a regular tag, treat it as a simple value
                self.processed_ifd[tag_name] = value


# Maps TIFF image tag numbers to the tag names.
TAGNUM2NAME = {
    11: 'ProcessingSoftware',
    254: 'NewSubfileType',
    255: 'SubfileType',
    256: 'ImageWidth',
    257: 'ImageLength',
    258: 'BitsPerSample',
    259: 'Compression',
    262: 'PhotometricInterpretation',
    263: 'Threshholding',
    264: 'CellWidth',
    265: 'CellLength',
    266: 'FillOrder',
    269: 'DocumentName',
    270: 'ImageDescription',
    271: 'Make',
    272: 'Model',
    273: 'StripOffsets',
    274: 'Orientation',
    277: 'SamplesPerPixel',
    278: 'RowsPerStrip',
    279: 'StripByteCounts',
    280: 'MinSampleValue',
    281: 'MaxSampleValue',
    282: 'XResolution',
    283: 'YResolution',
    284: 'PlanarConfiguration',
    286: 'XPosition',
    287: 'YPosition',
    290: 'GrayResponseUnit',
    291: 'GrayResponseCurve',
    292: 'T4Options',
    293: 'T6Options',
    296: 'ResolutionUnit',
    297: 'PageNumber',
    301: 'TransferFunction',
    305: 'Software',
    306: 'DateTime',
    315: 'Artist',
    316: 'HostComputer',
    317: 'Predictor',
    318: 'WhitePoint',
    319: 'PrimaryChromaticities',
    320: 'ColorMap',
    321: 'HalftoneHints',
    322: 'TileWidth',
    323: 'TileLength',
    324: 'TileOffsets',
    325: 'TileByteCounts',
    330: 'SubIFDs',
    332: 'InkSet',
    333: 'InkNames',
    334: 'NumberOfInks',
    336: 'DotRange',
    337: 'TargetPrinter',
    338: 'ExtraSamples',
    339: 'SampleFormat',
    340: 'SMinSampleValue',
    341: 'SMaxSampleValue',
    342: 'TransferRange',
    343: 'ClipPath',
    344: 'XClipPathUnits',
    345: 'YClipPathUnits',
    346: 'Indexed',
    347: 'JPEGTables',
    351: 'OPIProxy',
    512: 'JPEGProc',
    513: 'JPEGInterchangeFormat',
    514: 'JPEGInterchangeFormatLength',
    515: 'JPEGRestartInterval',
    517: 'JPEGLosslessPredictors',
    518: 'JPEGPointTransforms',
    519: 'JPEGQTables',
    520: 'JPEGDCTables',
    521: 'JPEGACTables',
    529: 'YCbCrCoefficients',
    530: 'YCbCrSubSampling',
    531: 'YCbCrPositioning',
    532: 'ReferenceBlackWhite',
    700: 'XMLPacket',
    18246: 'Rating',
    18249: 'RatingPercent',
    32781: 'ImageID',
    32996: 'Datatype',
    32997: 'ImageDepth',
    32998: 'TileDepth',
    33421: 'CFARepeatPatternDim',
    33422: 'CFAPattern',
    33423: 'BatteryLevel',
    33432: 'Copyright',
    33434: 'ExposureTime',
    33437: 'FNumber',
    33550: 'ModelPixelScale',
    33723: 'IPTCNAA',
    33918: 'INGRPacketDataTag',
    33922: 'ModelTiePoint',
    34264: 'ModelTransformation',
    34377: 'ImageResources',
    34665: 'ExifTag',
    34675: 'ICCProfile',
    34735: 'GeoKeyDirectory',
    34736: 'GeoDoubleParams',
    34737: 'GeoAsciiParams',
    34850: 'ExposureProgram',
    34852: 'SpectralSensitivity',
    34853: 'GPSTag',
    34855: 'ISOSpeedRatings',
    34856: 'OECF',
    34857: 'Interlace',
    34858: 'TimeZoneOffset',
    34859: 'SelfTimerMode',
    34864: 'SensitivityType',
    34865: 'StandardOutputSensitivity',
    34866: 'RecommendedExposureIndex',
    34867: 'ISOSpeed',
    34868: 'ISOSpeedLatitudeYYY',
    34869: 'ISOSpeedLatitudeZZZ',
    36864: 'ExifVersion',
    36880: 'OffsetTime',
    36881: 'OffsetTimeOriginal',
    36882: 'OffsetTimeDigitized',
    36867: 'DateTimeOriginal',
    36868: 'DateTimeDigitized',
    37121: 'ComponentsConfiguration',
    37122: 'CompressedBitsPerPixel',
    37377: 'ShutterSpeedValue',
    37378: 'ApertureValue',
    37379: 'BrightnessValue',
    37380: 'ExposureBiasValue',
    37381: 'MaxApertureValue',
    37382: 'SubjectDistance',
    37383: 'MeteringMode',
    37384: 'LightSource',
    37385: 'Flash',
    37386: 'FocalLength',
    37387: 'FlashEnergy',
    37388: 'SpatialFrequencyResponse',
    37389: 'Noise',
    37390: 'FocalPlaneXResolution',
    37391: 'FocalPlaneYResolution',
    37392: 'FocalPlaneResolutionUnit',
    37393: 'ImageNumber',
    37394: 'SecurityClassification',
    37395: 'ImageHistory',
    37396: 'SubjectLocation',
    37397: 'ExposureIndex',
    37398: 'TIFFEPStandardID',
    37399: 'SensingMethod',
    37500: 'MakerNote',
    37510: 'UserComment',
    37520: 'SubSecTime',
    37521: 'SubSecTimeOriginal',
    37522: 'SubSecTimeDigitized',
    37888: 'Temperature',
    37889: 'Humidity',
    37890: 'Pressure',
    37891: 'WaterDepth',
    37892: 'Acceleration',
    37893: 'CameraElevationAngle',
    40091: 'XPTitle',
    40092: 'XPComment',
    40093: 'XPAuthor',
    40094: 'XPKeywords',
    40095: 'XPSubject',
    40960: 'FlashPixVersion',
    40961: 'ColorSpace',
    40962: 'PixelXDimension',
    40963: 'PixelYDimension',
    41483: 'FlashEnergy',
    41484: 'SpatialFrequencyResponse',
    41486: 'FocalPlaneXResolution',
    41487: 'FocalPlaneYResolution',
    41488: 'FocalPlaneResolutionUnit',
    41492: 'SubjectLocation',
    41493: 'ExposureIndex',
    41495: 'SensingMethod',
    41728: 'FileSource',
    41729: 'SceneType',
    41730: 'CFAPattern',
    41985: 'CustomRendered',
    41986: 'ExposureMode',
    41987: 'WhiteBalance',
    41988: 'DigitalZoomRatio',
    41989: 'FocalLengthIn35MMFilm',
    41990: 'SceneCaptureType',
    41991: 'GainControl',
    41992: 'Contrast',
    41993: 'Saturation',
    41994: 'Sharpness',
    41995: 'DeviceSettingDescription',
    41996: 'SubjectDistanceRange',
    42016: 'ImageUniqueID',
    42032: 'CameraOwnerName',
    42033: 'BodySerialNumber',
    42034: 'LensSpecification',
    42035: 'LensMake',
    42036: 'LensModel',
    42037: 'LensSerialNumber',
    42080: 'CompositeImage',
    42081: 'SourceImageNumberOfCompositeImage',
    42082: 'SourceExposureTimeOfCompositeImage',
    42112: 'GDALMetadata',
    42113: 'GDALNoData',
    42240: 'Gamma',
    50341: 'PrintImageMatching',
    50706: 'DNGVersion',
    50707: 'DNGBackwardVersion',
    50708: 'UniqueCameraModel',
    50709: 'LocalizedCameraModel',
    50710: 'CFAPlaneColor',
    50711: 'CFALayout',
    50712: 'LinearizationTable',
    50713: 'BlackLevelRepeatDim',
    50714: 'BlackLevel',
    50715: 'BlackLevelDeltaH',
    50716: 'BlackLevelDeltaV',
    50717: 'WhiteLevel',
    50718: 'DefaultScale',
    50719: 'DefaultCropOrigin',
    50720: 'DefaultCropSize',
    50721: 'ColorMatrix1',
    50722: 'ColorMatrix2',
    50723: 'CameraCalibration1',
    50724: 'CameraCalibration2',
    50725: 'ReductionMatrix1',
    50726: 'ReductionMatrix2',
    50727: 'AnalogBalance',
    50728: 'AsShotNeutral',
    50729: 'AsShotWhiteXY',
    50730: 'BaselineExposure',
    50731: 'BaselineNoise',
    50732: 'BaselineSharpness',
    50733: 'BayerGreenSplit',
    50734: 'LinearResponseLimit',
    50735: 'CameraSerialNumber',
    50736: 'LensInfo',
    50737: 'ChromaBlurRadius',
    50738: 'AntiAliasStrength',
    50739: 'ShadowScale',
    50740: 'DNGPrivateData',
    50741: 'MakerNoteSafety',
    50778: 'CalibrationIlluminant1',
    50779: 'CalibrationIlluminant2',
    50780: 'BestQualityScale',
    50781: 'RawDataUniqueID',
    50827: 'OriginalRawFileName',
    50828: 'OriginalRawFileData',
    50829: 'ActiveArea',
    50830: 'MaskedAreas',
    50831: 'AsShotICCProfile',
    50832: 'AsShotPreProfileMatrix',
    50833: 'CurrentICCProfile',
    50834: 'CurrentPreProfileMatrix',
    50879: 'ColorimetricReference',
    50931: 'CameraCalibrationSignature',
    50932: 'ProfileCalibrationSignature',
    50934: 'AsShotProfileName',
    50935: 'NoiseReductionApplied',
    50936: 'ProfileName',
    50937: 'ProfileHueSatMapDims',
    50938: 'ProfileHueSatMapData1',
    50939: 'ProfileHueSatMapData2',
    50940: 'ProfileToneCurve',
    50941: 'ProfileEmbedPolicy',
    50942: 'ProfileCopyright',
    50964: 'ForwardMatrix1',
    50965: 'ForwardMatrix2',
    50966: 'PreviewApplicationName',
    50967: 'PreviewApplicationVersion',
    50968: 'PreviewSettingsName',
    50969: 'PreviewSettingsDigest',
    50970: 'PreviewColorSpace',
    50971: 'PreviewDateTime',
    50972: 'RawImageDigest',
    50973: 'OriginalRawFileDigest',
    50974: 'SubTileBlockSize',
    50975: 'RowInterleaveFactor',
    50981: 'ProfileLookTableDims',
    50982: 'ProfileLookTableData',
    51008: 'OpcodeList1',
    51009: 'OpcodeList2',
    51022: 'OpcodeList3',
    51041: 'NoiseProfile',
}

# maps the TIFF enumerated datatype to the corresponding structs datatype code,
# along with and data width
DATATYPE2FMT = {
    1: ('B', 1),
    2: ('B', 1),
    3: ('H', 2),
    4: ('I', 4),
    5: ('II', 8),
    7: ('B', 1),
    9: ('i', 4),
    10: ('ii', 8),
    11: ('f', 4),
    12: ('d', 8),
    13: ('I', 4),
    16: ('Q', 8),
    17: ('q', 8),
    18: ('Q', 8)
}

TIFFTYPE2NP = {
    1: np.ubyte,
    2: str,
    3: np.ushort,
    4: np.uint32,
    5: np.double,
    6: np.byte,
    7: np.ubyte,
    8: np.short,
    9: np.int32,
    10: np.double,
    11: np.double,
    12: np.double,
    13: np.uint32,
    16: np.uint64,
    17: np.int64,
    18: np.uint64,
}
