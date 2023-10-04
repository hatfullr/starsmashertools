import tarfile
import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.path
import starsmashertools.helpers.string
import copy

class CompressionTask(object):
    """
    Compress files into a tar ball or extract files from a tar ball created by
    this class. Can work in parallel.
    """

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def isCompressedFile(filename : str):
        """
        Check if the given filename was compressed by `~.compress`.

        Parameters
        ----------
        filename : str
            Name of the file to check.

        Returns
        -------
            `True` if the filename was compressed by `~.compress` and `False`
            otherwise.
        """
        if not starsmashertools.helpers.path.isfile(filename):
            raise FileNotFoundError(filename)
        method = CompressionTask.get_compression_method(filename)
        with tarfile.open(filename, 'r:'+method) as tar:
            try:
                tar.getmember('compression.sstools')
            except KeyError:
                return False
        return True

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_compression_method(filename : str):
        if not starsmashertools.helpers.path.isfile(filename):
            raise FileNotFoundError(filename)
        methods = CompressionTask.get_methods()
        for method in methods:
            if filename.endswith(method): return method

    @staticmethod
    def get_methods():
        """
        Return a list of compression methods available for use with
        `~.compress`.
        """
        methods = list(tarfile.TarFile.OPEN_METH.keys())
        if 'tar' in methods: methods.remove('tar')
        return methods

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def create_sstools_identity_file(directory : str):
        """
        Create a file named compression.sstools for identifying tar balls we
        created using `~.compress`.

        Parameters
        ----------
        directory : str
            The directory to create the compression.sstools file.

        Returns
        -------
        fname
            The path to the created compression.sstools file.

        See Also
        --------
        `~.compress`
        `~.decompress`
        """
        fname = starsmashertools.helpers.path.join(directory, 'compression.sstools')
        with starsmashertools.helpers.file.open(fname, 'w') as f:
            f.write("This file is used by starsmashertools to identify compressed archives it created. If you are reading this, then it probably means something went wrong during a compression. It is always safe to delete this file, but if it is removed from the tar ball it belongs to then starsmashertools might have trouble understanding which files to decompress. See the starsmashertools for more information: https://starsmashertools.readthedocs.io")
        return fname

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def compress(
            self,
            files : list | tuple,
            filename : str,
            method : str,
            delete : bool = True,
            delete_after : bool = True,
            verbose : bool = True,
    ):
        """
        Create a compressed tar ball of this CompressionTask's files, preserving
        file creation times.
        
        A special file named `compression.sstools` is added to the final 
        compressed file for identification later when using `~.decompress`.

        Parameters
        ----------
        files : list, tuple
            Paths to files to compress.

        filename : str
            Name of the resulting compressed file.

        method : str
            Compression method to be used. Must be one of the available methods
            returned from `~.get_methods`.

        delete : bool, default = True
            If `True`, the files which are compressed are deleted.

        delete_after : bool, default = True
            If `True`, compressed files are deleted after compression has
            completed. If `False`, files are deleted after compression has 
            completed. If `delete` is `False` this option is ignored.

        verbose : bool, default = True
            If `True`, debug messages are printed to the console.

        Returns
        -------
        None

        See Also
        --------
        `~.decompress`
        """
        methods = CompressionTask.get_methods()
        starsmashertools.helpers.argumentenforcer.enforcevalues({
            'method' : methods,
        })
        
        if starsmashertools.helpers.path.isfile(filename):
            raise FileExistsError("Cannot compress files because the tar ball already exists: '%s'" % filename)

        dirname = starsmashertools.helpers.path.dirname(filename)
        fname = CompressionTask.create_sstools_identity_file(dirname)
        files += [fname]
        
        with tarfile.open(filename, 'w:'+method) as tar:
            for f in files:
                if verbose: print("Adding '%s' to '%s'" % (f, filename))
                try:
                    tar.add(f, arcname=starsmashertools.helpers.path.relpath(f, dirname))
                except:
                    tar.extractall()
                    starsmashertools.helpers.path.remove(filename)
                    starsmashertools.helpers.path.remove(fname)
                    raise
                if delete and not delete_after:
                    starsmashertools.helpers.path.remove(f)
        # Remove the old files
        if delete and delete_after:
            for f in files:
                starsmashertools.helpers.path.remove(f)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def decompress(
            self,
            filename : str,
            delete : bool = True,
            verbose : bool = False,
    ):
        """
        Decompress a tar ball that was created with `~.compress`.

        Parameters
        ----------
        filename : str
            The filename to decompress.
        
        delete : bool, default = True
            If `True`, the compressed file is deleted after decompression.
        
        verbose : bool, default = False
            If `True`, debug messages are printed to the console.

        Returns
        -------
        None

        See Also
        --------
        `~.compress`
        """

        method = CompressionTask.get_compression_method(filename)
        if method is None:
            raise NotImplementedError("Compression method not recognized for file '%s'. Expected the file name to end with one of %s" % (filename, starsmashertools.helpers.string.list_to_string(CompressionTask.get_methods(), join='or')))

        # Check to make sure this is one we compressed
        if not CompressionTask.isCompressedFile(filename):
            raise Exception("The given file is not a file created by CompressionTask because it is missing a 'compression.sstools' file. '%s'" % filename)

        dirname = starsmashertools.helpers.path.dirname(filename)
            
        extracted = []
        with tarfile.open(filename, 'r:'+method) as tar:
            for member in tar.getmembers():
                fname = starsmashertools.helpers.path.join(dirname, member.name)
                if starsmashertools.helpers.path.isfile(fname):
                    for e in extracted: starsmashertools.helpers.path.remove(e)
                    raise FileExistsError(fname)
                if verbose: print("Extracting '%s' to '%s'" % (fname, dirname))
                try:
                    tar.extract(member, path=dirname)
                except:
                    if starsmashertools.helpers.path.isfile(fname):
                        starsmashertools.helpers.path.remove(fname)
                    for e in extracted:
                        starsmashertools.helpers.path.remove(e)
                    raise
                if member.name == 'compression.sstools':
                    starsmashertools.helpers.path.remove(fname)
                else:
                    extracted += [fname]
        
        # Remove the tar ball
        if delete:
            starsmashertools.helpers.path.remove(filename)
        
