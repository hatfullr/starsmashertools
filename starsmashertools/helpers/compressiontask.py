import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.path
import starsmashertools.helpers.string
import starsmashertools.helpers.jsonfile
import multiprocessing
import time
import numpy as np
import zipfile
import datetime
import typing

class CompressionTask(object):
    _processes = []
    _pool = None

    @staticmethod
    def _create_pool():
        if CompressionTask._pool is not None or CompressionTask._processes:
            raise Exception("Cannot create a pool because a pool was already created")
        CompressionTask._pool = multiprocessing.Pool()
        CompressionTask._processes = [None]*CompressionTask._pool._processes

    @staticmethod
    def _close_pool():
        if CompressionTask._pool is None:
            raise Exception("Cannot close the pool because a pool was never created")
        CompressionTask._pool.close()
        CompressionTask._pool.join()
        CompressionTask._pool = None
        CompressionTask._processes = []

    @staticmethod
    def _assign_process(index, *args, **kwargs):
        CompressionTask._processes[index] = CompressionTask._pool.apply_async(*args, **kwargs)

    @staticmethod
    def _get_processes():
        results = [None] * len(CompressionTask._processes)
        for i, proc in enumerate(CompressionTask._processes):
            if proc is not None:
                results[i] = proc.get()
        return results

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_compressed_properties(filename : str):
        r"""
        Get a dictionary of properties on the files contained in the compressed
        archive.

        Parameters
        ----------
        filename : str
            The path to the compressed archive.

        Returns
        -------
        dict
            A dictionary whose keys are the names of the files in the compressed
            archive that they would have if the archive were decompressed. Each
            value is a dictionary holding various values corresponding to each
            file in the archive.
        """
        with zipfile.ZipFile(filename, 'r') as zfile:
            directory = starsmashertools.helpers.path.dirname(zfile.filename)
            identifier = CompressionTask.get_compression_identifier(zfile)
            return CompressionTask.get_properties_from_identifier(identifier, directory)

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_compression_identifier(zfile : zipfile.ZipFile):
        directory = starsmashertools.helpers.path.dirname(zfile.filename)
        filename = CompressionTask.get_compression_filename(zfile)
        arcname = CompressionTask._filename_to_arcname(filename, directory)
        content = zfile.read(arcname)
        return starsmashertools.helpers.jsonfile.load_bytes(content)

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_from_identifier(keys : list, identifier : dict):
        def do(i, key, result=[]):
            for _i, ident in enumerate(i['identifiers']):
                if ident is None:
                    result += [i[key][_i]]
                else:
                    do(ident, key, result=result)
            return result
        results = {key:do(identifier, key, result=[]) for key in keys}
        return results

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_properties_from_identifier(identifier : dict, directory : str):
        keys = list(identifier.keys())
        keys.remove('identifiers')
        properties = CompressionTask.get_from_identifier(keys, identifier)
        if 'arcnames' in properties.keys():
            properties['filenames'] = [CompressionTask._arcname_to_filename(arcname, directory) for arcname in properties['arcnames']]
        return properties
        
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def isCompressedFile(filename : str):
        r"""
        Check if the given filename was compressed by :meth:`~.compress`\.

        Parameters
        ----------
        filename : str
            Name of the file to check.

        Returns
        -------
            `True` if the filename was compressed by :meth:`~.compress` and
            `False` otherwise.
        """
        if not starsmashertools.helpers.path.isfile(filename):
            raise FileNotFoundError(filename)
        if not zipfile.is_zipfile(filename): return False
        with zipfile.ZipFile(filename, mode='r') as zfile:
            fname = CompressionTask.get_compression_filename(zfile)
            compression_filename = CompressionTask._filename_to_arcname(
                fname,
                starsmashertools.helpers.path.dirname(fname),
            )
            try:
                zfile.getinfo(compression_filename)
            except KeyError:
                return False
        return True
    

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def _filename_to_arcname(
            filename : str,
            directory : str,
    ):
        return starsmashertools.helpers.path.relpath(filename, directory)

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def _arcname_to_filename(
            arcname : str,
            directory : str,
    ):
        return starsmashertools.helpers.path.join(directory, arcname)
        
    @staticmethod
    def _pack_compression_file(
            zfile : zipfile.ZipFile,
            files : list | tuple,
    ):
        r"""
        Create the compression file identifier and add it to the given ZipFile.
        Delete the original file afterwards.
        """
        dirname = starsmashertools.helpers.path.dirname(zfile.filename)
        obj = {
            'arcnames' : [],
            'identifiers' : [],
            'mtimes' : [],
        }
        for _file in files:
            arcname = CompressionTask._filename_to_arcname(_file, dirname)
            obj['arcnames'] += [arcname]
            obj['mtimes'] += [starsmashertools.helpers.path.getmtime(_file)]
            if CompressionTask.isCompressedFile(_file):
                with zipfile.ZipFile(_file, mode='r') as _zfile:
                    obj['identifiers'] += [CompressionTask.get_compression_identifier(_zfile)]
            else: obj['identifiers'] += [None]
        
        ssfilename = CompressionTask.get_compression_filename(zfile)
        
        # Create the sstools_compression.json file
        starsmashertools.helpers.jsonfile.save(ssfilename, obj)
        
        arcname = CompressionTask._filename_to_arcname(ssfilename, dirname)
        zfile.write(ssfilename, arcname=arcname)
        starsmashertools.helpers.path.remove(ssfilename)

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def _unpack_compression_file(zfile : zipfile.ZipFile):
        dirname = starsmashertools.helpers.path.dirname(zfile.filename)
        ssfilename = CompressionTask.get_compression_filename(zfile)
        
        todecompress = []
        obj = starsmashertools.helpers.jsonfile.load(ssfilename)
        for arcname in obj['arcnames']:
            _path = CompressionTask._arcname_to_filename(arcname, dirname)
            if CompressionTask.isCompressedFile(_path):
                todecompress += [_path]
        starsmashertools.helpers.path.remove(ssfilename)

        for _file in todecompress:
            CompressionTask.decompress(_file)

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_compression_filename(zfile : zipfile.ZipFile):
        return zfile.filename + '.json'
    
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def compress(
            files : list | tuple,
            filename : str,
            delete : bool = True,
            delete_after : bool = True,
            verbose : typing.Callable | bool = True,
            nprocs : int = 0,
    ):
        r"""
        Create a compressed zip archive of this CompressionTask's files,
        preserving file creation times.
        
        A special file is added to the final compressed file for identification 
        later when using :meth:`~.decompress`\.
        
        Parameters
        ----------
        files : list, tuple
            Paths to files to compress.

        filename : str
            Name of the resulting compressed file.

        delete : bool, default = True
            If `True`\, the files which are compressed are deleted.

        delete_after : bool, default = True
            If `True`\, compressed files are deleted only after all files have
            been compressed. If `False`\, each file is deleted after it has been
            compressed. If `delete` is `False` this option is ignored.

        verbose : callable, bool, default = True
            Determines if the compressed file names should be printed to the
            console when they are about to be compressed.

            If `True`\, debug messages are printed to the console. If a callable
            function is given, it is called with a string passed as the first
            argument instead of printing to the console.

        nprocs : int, default = 0
            Use this many processes to perform the compression. A value of 0
            means to use as many processes as possible.

        See Also
        --------
        :meth:`~.compress_serial`
        :meth:`~.compress_parallel`
        :meth:`~.decompress`
        """
        
        if nprocs == 0:
            nprocs = multiprocessing.cpu_count()
            
        if nprocs > 1:
            CompressionTask.compress_parallel(
                files,
                filename,
                delete=delete,
                delete_after=delete_after,
                verbose=verbose,
            )
        else:
            CompressionTask.compress_serial(
                files,
                filename,
                delete=delete,
                delete_after=delete_after,
                verbose=verbose,
            )
        


            
            
    @staticmethod
    def compress_serial(
            files : list | tuple,
            filename : str,
            delete : bool = True,
            delete_after : bool = True,
            verbose : typing.Callable | bool = True,
    ):
        r"""
        Perform compression in serial mode.

        Parameters
        ----------
        files : list, tuple
            Paths to files to compress.

        filename : str
            Name of the resulting compressed file.

        delete : bool, default = True
            If `True`\, the files which are compressed are deleted.

        delete_after : bool, default = True
            If `True`\, compressed files are deleted only after all files have
            been compressed. If `False`\, each file is deleted after it has been
            compressed. If `delete` is `False` this option is ignored.

        verbose : callable, bool, default = True
            Determines if the compressed file names should be printed to the
            console when they are about to be compressed.
            
            If `True`\, messages are printed to the console. If a callable
            function is given, it is called with a string passed as the first
            argument instead of printing to the console.

        See Also
        --------
        :meth:`~.compress_parallel`
        :meth:`~.decompress`
        """
        
        if starsmashertools.helpers.path.isfile(filename):
            raise FileExistsError("Cannot compress simulation because it is already compressed at '%s'" % filename)

        if isinstance(verbose, bool) and verbose:
            verbose = lambda message: print(message)

        dirname = starsmashertools.helpers.path.dirname(filename)
        try:
            with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zfile:

                CompressionTask._pack_compression_file(zfile, files)
            
                for _file in files:
                    if verbose: verbose(_file)
                    arcname = CompressionTask._filename_to_arcname(_file, dirname)
                    zinfo = zipfile.ZipInfo.from_file(_file, arcname=arcname)
                    zinfo.compress_type = zipfile.ZIP_DEFLATED
                    zinfo.date_time = tuple(list(datetime.datetime.fromtimestamp(
                        starsmashertools.helpers.path.getmtime(_file)
                    ).timetuple())[:6])
                    with starsmashertools.helpers.file.open(_file, 'rb') as f:
                        zfile.writestr(zinfo, f.read())

                    if delete and not delete_after:
                        starsmashertools.helpers.path.remove(_file)
        except:
            if starsmashertools.helpers.path.isfile(filename):
                with zipfile.ZipFile(filename, mode='r') as zfile:
                    zfile.extractall(path=dirname)
                starsmashertools.helpers.path.remove(filename)
            raise

        # Remove the old files
        if delete and delete_after:
            for f in files:
                starsmashertools.helpers.path.remove(f)


    
                
    
    @staticmethod
    def compress_parallel(
            files : list | tuple,
            filename : str,
            **kwargs
    ):
        r"""
        Perform compression in parallel mode. The list of files is divvied up
        among processes, who each do :meth:`~.compress_serial`\. When all processes
        are finished each zip file is added to the final main zip file.

        Parameters
        ----------
        files : list, tuple
            Paths to files to compress.

        filename : str
            Name of the resulting compressed file.

        Other Parameters
        ----------------
        kwargs : dict
            Other keyword arguments are passed directly to
            :meth:`~.compress_serial`\.

        See Also
        --------
        :meth:`~.compress_serial`
        """

        if starsmashertools.helpers.path.isfile(filename):
            raise FileExistsError("Cannot compress simulation because it is already compressed at '%s'" % filename)

        CompressionTask._create_pool()
        
        chunks = np.array_split(files, len(CompressionTask._processes))
        filenames = []
        for i, chunk in enumerate(chunks):
            if len(chunk) == 0: continue
            fname = filename+"."+str(i)
            filenames += [fname]
            CompressionTask._assign_process(
                i,
                CompressionTask.compress_serial,
                args=(
                    chunk,
                    fname,
                ),
                kwds=kwargs,
            )
        
        CompressionTask._get_processes()
        CompressionTask._close_pool()

        files_done = [starsmashertools.helpers.path.isfile(f) for f in filenames]
        
        # After the individual processes finish, we compress the individual zip
        # files into the final zip file
        CompressionTask.compress_serial(
            filenames,
            filename,
            delete = True,
            delete_after = True,
            verbose = False,
        )
            
        

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def decompress(
            filename : str,
            delete : bool = True,
            verbose : typing.Callable | bool = False,
    ):
        r"""
        Decompress a file created by :meth:`~.compress`\.

        Parameters
        ----------
        filename : str
            The filename to decompress.
        
        delete : bool, default = True
            If `True`\, the compressed file is deleted after decompression.
        
        verbose : callable, bool, default = True
            Determines if the decompressed file names should be printed to the
            console when they are about to be decompressed.
            
            If `True`\, debug messages are printed to the console. If a callable
            function is given, it is called with a string passed as the first
            argument instead of printing to the console.
        
        See Also
        --------
        :meth:`~.compress`
        """

        if not starsmashertools.helpers.path.isfile(filename):
            raise FileNotFoundError("Cannot decompress simulation because the corresponding compression file was not found: '%s'" % filename)
        
        # Check to make sure this is one we compressed
        if not CompressionTask.isCompressedFile(filename):
            raise Exception("The given file is not a file created by CompressionTask because it is missing the compression identifier file: '%s'" % filename)

        if isinstance(verbose, bool) and verbose:
            verbose = lambda message: print(message)
        
        dirname = starsmashertools.helpers.path.dirname(filename)
        
        with zipfile.ZipFile(filename) as zfile:
            for zinfo in zfile.infolist():
                fname = CompressionTask._arcname_to_filename(zinfo.filename, dirname)
                if verbose: verbose(fname)
                
                # Note: extract does not remove the file from the zip archive.
                zfile.extract(zinfo, path=dirname)

                mtime = datetime.datetime(*zinfo.date_time).timestamp()
                # Note: this mtime is accurate only up to the decimal. The
                # zipfile module does not support additional accuracy...
                starsmashertools.helpers.path.utime(fname, times=(time.time(), mtime))
            CompressionTask._unpack_compression_file(zfile)
        
        # Remove the zip archive
        if delete:
            starsmashertools.helpers.path.remove(filename)



        
