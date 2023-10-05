import tarfile
import gzip
import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.path
import starsmashertools.helpers.string
import copy
import multiprocessing
import time
import shutil
import numpy as np

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
        if starsmashertools.helpers.path.isfile(fname):
            raise FileExistsError(fname)
        
        with starsmashertools.helpers.file.open(fname, 'w') as f:
            f.write("This file is used by starsmashertools to identify compressed archives it created. If you are reading this, then it probably means something went wrong during a compression. It is always safe to delete this file, but if it is removed from the tar ball it belongs to then starsmashertools might have trouble understanding which files to decompress. See the starsmashertools for more information: https://starsmashertools.readthedocs.io")
        return fname




    @staticmethod
    def _gzip_compress(filenames, delete):
        # Return a list of compressed filenames. If delete is True then
        # we delete the original files immediately after compressing them
        ret = []
        for filename in filenames:
            mtime = starsmashertools.helpers.path.getmtime(filename)
            new_filename = filename+".gz"
            ret += [new_filename]
            try:
                with open(filename, 'rb') as f_in:
                    with gzip.open(new_filename, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                starsmashertools.helpers.path.utime(new_filename, times=(time.time(), mtime))
            except:
                if delete: # Decompress instead of delete
                    CompressionTask._gzip_decompress(ret)
                else:
                    # User still has the original files, so we only need to
                    # delete the ones we already made
                    for r in ret:
                        if starsmashertools.helpers.path.isfile(r):
                            starsmashertools.helpers.path.remove(r)
                raise
            if delete:
                starsmashertools.helpers.path.remove(filename)
        return ret
    
    @staticmethod
    def _gzip_decompress(filenames):
        # Return a list of decompressed filenames and delete the compressed
        # files.
        ret = []
        for filename in filenames:
            if not filename.endswith('.gz'):
                raise Exception("Cannot decompress file that isn't a gzip file: '%s'" % filename)
            new_filename = filename[:-len('.gz')]
            with gzip.open(filename, 'rb') as f_in:
                with open(new_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            starsmashertools.helpers.path.utime(
                new_filename,
                times=(
                    time.time(),
                    starsmashertools.helpers.path.getmtime(filename),
                ),
            )
            starsmashertools.helpers.path.remove(filename)
    

    

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def compress(
            *args,
            **kwargs):
        """
        Create a compressed tar ball of this CompressionTask's files, preserving
        file creation times.
        
        A special file named `compression.sstools` is added to the final 
        compressed file for identification later when using `~.decompress`.

        Parameters
        ----------
        *args
            Arguments are passed directly to either `~.compress_serial` or
            `~.compress_parallel` depending on the system.
        
        **kwargs
            Additional keyword arguments are passed directly to either
            `~.compress_serial` or `~.compress_parallel` depending on the
            system.
        

        See Also
        --------
        `~.compress_serial`
        `~.compress_parallel`
        `~.decompress`
        """
        parallel = kwargs.pop('parallel', None)
        if parallel is None:
            nprocs = multiprocessing.cpu_count()
            parallel = nprocs > 1
        if parallel:
            CompressionTask.compress_parallel(*args, **kwargs)
        else:
            CompressionTask.compress_serial(*args, **kwargs)
        
            
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def compress_serial(
            files : list | tuple,
            filename : str,
            method : str,
            delete : bool = True,
            delete_after : bool = True,
            verbose : bool = True,
    ):
        """
        Perform compression in serial mode.

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
            If `True`, compressed files are deleted only after all files have
            been compressed. If `False`, each file is deleted after it has been
            compressed. If `delete` is `False` this option is ignored.

        verbose : bool, default = True
            If `True`, debug messages are printed to the console.

        See Also
        --------
        `~.compress_parallel`
        `~.decompress_serial`
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

    
                
    
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def compress_parallel(
            files : list | tuple,
            filename : str,
            method : str,
            delete : bool = True,
            delete_after : bool = True,
            verbose : bool = True,
    ):
        """
        Perform compression in parallel mode. Processes are spawned and each one
        compresses a single file at a time using gzip. The compressed files are
        then each added to 

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
            If `True`, compressed files are deleted only after all files have
            been compressed. If `False`, each file is deleted after it has been
            compressed. If `delete` is `False` this option is ignored.
        
        verbose : bool, default = True
            If `True`, debug messages are printed to the console.

        See Also
        --------
        `~.compress_serial`
        `~.decompress_parallel`
        """
        methods = CompressionTask.get_methods()
        starsmashertools.helpers.argumentenforcer.enforcevalues({
            'method' : methods,
        })
        
        if starsmashertools.helpers.path.isfile(filename):
            raise FileExistsError("Cannot compress files because the tar ball already exists: '%s'" % filename)

        pool = multiprocessing.Pool()
        nprocs = pool._processes
        
        chunks = np.array_split(files, nprocs)
        processes = [None]*nprocs
        for i in range(nprocs):
            processes[i] = pool.apply_async(
                CompressionTask._gzip_compress,
                args = (chunks[i], delete,),
            )
        
        results = [None]*nprocs
        for i in range(nprocs):
            results[i] = processes[i].get()

        pool.close()
        pool.join()

        compressed_filenames = []
        for result in results:
            compressed_filenames += result
            
        dirname = starsmashertools.helpers.path.dirname(filename)
        
        # Add all the compressed files to the tar ball
        added_files = []
        with tarfile.open(filename, 'w:'+method) as tar:
            for f in compressed_filenames:
                try:
                    tar.add(f, arcname=starsmashertools.helpers.path.relpath(f, dirname))
                    added_files += [f]
                except:
                    tar.extractall()
                    # If the user chose to delete files as they are compressed
                    # then we need to restore the original files from the
                    # compressed ones
                    CompressionTask._gzip_decompress(added_files)
                    raise

            # Finally create and add the compression.sstools file
            fname = CompressionTask.create_sstools_identity_file(dirname)
            try:
                tar.add(fname, arcname=starsmashertools.helpers.path.relpath(fname, dirname))
            except:
                tar.extractall()
                starsmashertools.helpers.path.remove(fname)
                raise
            starsmashertools.helpers.path.remove(fname)
        
        for f in compressed_filenames:
            starsmashertools.helpers.path.remove(f)
        

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def decompress(
            *args,
            **kwargs
    ):
        """
        Decompress a tar ball that was created with `~.compress`.

        Parameters
        ----------
        *args
            Arguments are passed directly to either `~.decompress_serial` or
            `~.decompress_parallel` depending on the system.
        
        **kwargs
            Additional keyword arguments are passed directly to either 
            `~.decompress_serial` or `~.decompress_parallel` depending on the
            system.
        
        See Also
        --------
        `~.decompress_serial`
        `~.decompress_parallel`
        `~.compress`
        """
        parallel = kwargs.pop('parallel', None)
        if parallel is None:
            nprocs = multiprocessing.cpu_count()
            parallel = nprocs > 1

        if parallel:
            CompressionTask.decompress_parallel(*args, **kwargs)
        else:
            CompressionTask.decompress_serial(*args, **kwargs)
    
        
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def decompress_serial(
            filename : str,
            delete : bool = True,
            verbose : bool = False,
    ):
        """
        Decompress a tar ball that was created with `~.compress` in serial.

        Parameters
        ----------
        filename : str
            The filename to decompress.
        
        delete : bool, default = True
            If `True`, the compressed file is deleted after decompression.
        
        verbose : bool, default = False
            If `True`, debug messages are printed to the console.

        See Also
        --------
        `~.decompress_parallel`
        `~.compress_serial`
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

    
    
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def decompress_parallel(
            filename : str,
            delete : bool = True,
            verbose : bool = False,
    ):
        """
        Decompress a tar ball that was created with `~.compress` in parallel.

        Parameters
        ----------
        filename : str
            The filename to decompress.
        
        delete : bool, default = True
            If `True`, the compressed file is deleted after decompression.
        
        verbose : bool, default = False
            If `True`, debug messages are printed to the console.

        See Also
        --------
        `~.decompress_serial`
        `~.compress_parallel`
        """

        method = CompressionTask.get_compression_method(filename)
        if method is None:
            raise NotImplementedError("Compression method not recognized for file '%s'. Expected the file name to end with one of %s" % (filename, starsmashertools.helpers.string.list_to_string(CompressionTask.get_methods(), join='or')))
        
        # Check to make sure this is one we compressed
        if not CompressionTask.isCompressedFile(filename):
            raise Exception("The given file is not a file created by CompressionTask because it is missing a 'compression.sstools' file. '%s'" % filename)
        
        dirname = starsmashertools.helpers.path.dirname(filename)

        # Unpack the tar ball first
        extracted = []
        with tarfile.open(filename, 'r:'+method) as tar:
            for member in tar.getmembers():
                fname = starsmashertools.helpers.path.join(dirname, member.name)

                if member.name == 'compression.sstools':
                    tar.extract(member, path=dirname)
                    starsmashertools.helpers.path.remove(fname)
                    continue
                
                if not fname.endswith('.gz'):
                    raise Exception("Found a file that was not compressed with gzip: '%s'" % fname)
                
                if starsmashertools.helpers.path.isfile(fname[:-len('.gz')]):
                    for e in extracted: starsmashertools.helpers.path.remove(e)
                    raise FileExistsError(fname[:-len('.gz')])

                
                if verbose: print("Extracting '%s' to '%s'" % (fname, dirname))
                
                try:
                    tar.extract(member, path=dirname)
                    extracted += [fname]
                except:
                    if starsmashertools.helpers.path.isfile(fname):
                        starsmashertools.helpers.path.remove(fname)
                    for e in extracted:
                        starsmashertools.helpers.path.remove(e)
                    raise
        
        # Delete the tar ball
        if delete:
            starsmashertools.helpers.path.remove(filename)

        # Now split up processes to decompress each file
        pool = multiprocessing.Pool()
        nprocs = pool._processes
        
        chunks = np.array_split(extracted, nprocs)
        processes = [None]*nprocs
        for i in range(nprocs):
            processes[i] = pool.apply_async(
                CompressionTask._gzip_decompress,
                args = (chunks[i],),
            )
        
        results = [None]*nprocs
        for i in range(nprocs):
            results[i] = processes[i].get()

        pool.close()
        pool.join()
