-------------------------------------
Platforms Tested (0.7.0 release only)
-------------------------------------
    * Linux Mint 17 / Python 3.4.0 and 2.7.6 / OpenJPEG 2.1.0
    * MacOS 10.6.8 / MacPorts / Python 3.4.1
    * CentOS 6.5 / Anaconda Python 3.4.1 / OpenJPEG 1.3
      (please use OpenJPEG 2.1.0 instead, though)

------------
Known Issues
------------

    * Creating a Jp2 file with the irreversible option does not work
      on windows.
    * Eval-ing a :py:meth:`repr` string does not work on windows.

-------
Roadmap
-------

Here's an incomplete list of what I'd like to focus on in the future.

    * continue to monitor upstream changes in the openjp2 library
    * investigate JPIP (likely to be a big project)
