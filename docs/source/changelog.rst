---------
ChangeLog
---------

0.6.0 (pending)
===============

      * Added Cinema2K, Cinema4K write support.
      * Added lxml requirement.
      * added set_printoptions, get_printoptions function
      * dropped support for Python 2.6, added support for Python 3.4
      * dropped support for OpenJPEG versions 1.3 and 1.4
      * dropped windows support (it might work, it might not, I don't much care)
      * added write support for JP2 UUID, dataEntryURL, palette, and component mapping boxes
      * added read/write support for JPX free, number list, and data reference boxes
      * Added read support for JPX fragment list and fragment table boxes
      * incompatible change to channel definition box constructor, channel_type and association are no longer keyword arguments
      * incompatible change to palette box constructor, it now takes a 2D numpy array instead of a list of 1D arrays

0.5.0 (September 16, 2013)
==========================
    
      * added write support when using OpenJPEG version 1.5
      * added version module

0.4.0 (August 18, 2013)
==========================
    
      * added append method

0.3.0 (July 31, 2013)
==========================
    
      * added support for OpenJPEG library version 2.0.0

0.2.0 (July 11, 2013)
==========================
    
      * added Python 2.6, Python 2.7 on windows
      * read/write using OpenJPEG library version 2.0.0
      * read using OpenJPEG 1.4

0.1.0 (May 27, 2013)
====================
    
      * first release using development version (2.x) of OpenJPEG
