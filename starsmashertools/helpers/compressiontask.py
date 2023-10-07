import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.path
import starsmashertools.helpers.string
import starsmashertools.helpers.jsonfile
import copy
import multiprocessing
import time
import shutil
import numpy as np
import zipfile


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
        if not zipfile.is_zipfile(filename): return False
        zfile = zipfile.ZipFile(filename, mode='r')
        try:
            zfile.getinfo('sstools_compression.json')
        except KeyError:
            return False
        return True

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
    def compress(
            *args,
            **kwargs):
        """
        Create a compressed tar ball of this CompressionTask's files, preserving
        file creation times.
        
        A special file named `sstools_compression.json` is added to the final 
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
        #if parallel is None:
        #    nprocs = multiprocessing.cpu_count()
        #    parallel = nprocs > 1
        #if parallel:
        #    CompressionTask.compress_parallel(files, *args, **kwargs)
        #else:
        #    CompressionTask.compress_serial(files, *args, **kwargs)
        CompressionTask.compress_serial(*args, **kwargs)
            
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def compress_serial(
            files : list | tuple,
            filename : str,
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
        
        if starsmashertools.helpers.path.isfile(filename):
            raise FileExistsError(filename)

        obj = {}
        
        dirname = starsmashertools.helpers.path.dirname(filename)
        with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zfile:
            arcnames = []
            for _file in files:
                arcname = starsmashertools.helpers.path.relpath(_file, dirname)
                obj[arcname] = {
                    'mtime' : starsmashertools.helpers.path.getmtime(_file),
                }

                zfile.write(_file, arcname=arcname)
                
                if delete and not delete_after:
                    starsmashertools.helpers.path.remove(_file)
            
            ssfilename = starsmashertools.helpers.path.join(
                dirname,
                'sstools_compression.json',
            )
            
            # Create the sstools_compression.json file
            starsmashertools.helpers.jsonfile.save(ssfilename, obj)

            arcname = starsmashertools.helpers.path.relpath(ssfilename, dirname)
            zfile.write(ssfilename, arcname=arcname)
        
        # Remove the old files
        starsmashertools.helpers.path.remove(ssfilename)
        if delete and delete_after:
            for f in files:
                starsmashertools.helpers.path.remove(f)

        

    
                
    
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def compress_parallel(
            files : list | tuple,
            filename : str,
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
        """
        if starsmashertools.helpers.path.isfile(filename):
            raise FileExistsError("Cannot compress files because the tar ball already exists: '%s'" % filename)

        directory = starsmashertools.helpers.path.dirname(filename)
        fname = starsmashertools.helpers.path.join(directory, 'sstools_compression.json')
        tocompress = copy.deepcopy(files)
        toadd = []
        if starsmashertools.helpers.path.isfile(fname):
            with starsmashertools.helpers.file.open(fname, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0 and line[0] == '#': continue
                    if starsmashertools.helpers.path.isfile(line):
                        toadd += [line]
                        tocompress.remove(line)
        
        pool = multiprocessing.Pool()
        nprocs = pool._processes
        
        chunks = np.array_split(tocompress, nprocs)
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
            # Add files that were previously compressed but for some reason not
            # added to the tar ball
            for f in toadd:
                try:
                    tar.add(f, arcname=starsmashertools.helpers.path.relpath(f, dirname))
                    added_files += [f]
                except:
                    tar.extractall()
                    CompressionTask._gzip_decompress(added_files)
                    raise
            
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

            # Finally create and add the sstools_compression.json file
            if starsmashertools.helpers.path.isfile(fname):
                with starsmashertools.helpers.file.open(fname, 'a') as f:
                    for _file in compressed_filenames:
                        f.write(_file+"\n")
            else:
                CompressionTask.create_sstools_identity_file(fname, added_files)
                
            try:
                tar.add(fname, arcname=starsmashertools.helpers.path.relpath(fname, dirname))
            except:
                tar.extractall()
                starsmashertools.helpers.path.remove(fname)
                raise
            starsmashertools.helpers.path.remove(fname)
        
        for f in compressed_filenames:
            if starsmashertools.helpers.path.isfile(f):
                starsmashertools.helpers.path.remove(f)
        """

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
        #if parallel is None:
        #    nprocs = multiprocessing.cpu_count()
        #    parallel = nprocs > 1
        #
        #if parallel:
        #    CompressionTask.decompress_parallel(*args, **kwargs)
        #else:
        #    CompressionTask.decompress_serial(*args, **kwargs)
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
        
        
        # Check to make sure this is one we compressed
        if not CompressionTask.isCompressedFile(filename):
            raise Exception("The given file is not a file created by CompressionTask because it is missing a 'sstools_compression.json' file. '%s'" % filename)
        
        dirname = starsmashertools.helpers.path.dirname(filename)
        
        with zipfile.ZipFile(filename) as zfile:
            zfile.extractall(path=dirname)

        ssfilename = starsmashertools.helpers.path.join(
            dirname,
            'sstools_compression.json',
        )
        
        obj = starsmashertools.helpers.jsonfile.load(ssfilename)
        for relpath, stuff in obj.items():
            _path = starsmashertools.helpers.path.join(dirname, relpath)
            starsmashertools.helpers.path.utime(_path, times=(time.time(), stuff['mtime']))
        starsmashertools.helpers.path.remove(ssfilename)
            
        # Remove the zip archive
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
        """
        method = CompressionTask.get_compression_method(filename)
        if method is None:
            raise NotImplementedError("Compression method not recognized for file '%s'. Expected the file name to end with one of %s" % (filename, starsmashertools.helpers.string.list_to_string(CompressionTask.get_methods(), join='or')))
        
        # Check to make sure this is one we compressed
        if not CompressionTask.isCompressedFile(filename):
            raise Exception("The given file is not a file created by CompressionTask because it is missing a 'sstools_compression.json' file. '%s'" % filename)
        
        dirname = starsmashertools.helpers.path.dirname(filename)

        # Unpack the tar ball first
        extracted = []
        with tarfile.open(filename, 'r:'+method) as tar:
            for member in tar.getmembers():
                fname = starsmashertools.helpers.path.join(dirname, member.name)

                if member.name == 'sstools_compression.json':
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
        """
