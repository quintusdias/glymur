------------
Known Issues
------------

    * WinPython 3.3.2 and OpenJPEG 2.0 don't seem to want to play well together.  If you do not need write support, just use OpenJPEG 1.5 instead.  If you do need write support, try the development version of OpenJPEG.

-------
Roadmap
-------

Here's an incomplete list of what I'd like to focus on in the future.

    * continue to monitor upstream changes in the openjp2 library
    * investigate using CFFI or cython instead of ctypes to wrap openjp2
    * investigate adding write support for UUID/XMP boxes (potentially a big project)
    * investigate JPIP (likely to be an even bigger project)
    
