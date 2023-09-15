      subroutine init_cooling
      include '../starsmasher.h'
      include 'cooling.h'
      integer i, j, ncols, status
      character*255 fmt, fmt2, fmt3
      character*16 firstPart,expectedFirstPart
      parameter(expectedFirstPart="      logT,logR ")
      real*8 logR(1000), logT(1000)
      common/lrt/ logR, logT
      real*8 opacityTable(1000,1000)
      common/opacitytablecomm/opacityTable

      character*17000 line
      
      integer testRows, testCols
      real*8 testlogRmin, testlogTmin, testdlogR, testdlogT,
     $     testlogR, testlogT, getOpacity, testRho, testT
      character*1028 dummy
      real*8 theta,phi

      integer debug
      parameter(debug=0)
      
      if(myrank.eq.0) write(*,*) "init_cooling started"
      
      crad_codeunits = crad / sqrt(gravconst*munit/runit)
      arad_codeunits = arad / (gravconst*munit**2/runit**4)
      sigma_codeunits = 0.25d0 * crad_codeunits * arad_codeunits

      if(myrank.eq.0) then
         write(69,*) "Using crad_codeunits, arad_codeunits = ",
     $        crad_codeunits, arad_codeunits
      end if
      
c     Initialize the icosahedron angles
      call init_icosahedron(icosahedron_subdivisions)
      
      do i=1, n
         popacity(i) = 0.d0
         temperatures(i) = 0.d0
      end do
      
c     Read in the opacity table
      open(44,file=trim(adjustl(opacityfile)),status='old')

      read(44,'(A)') firstPart
      if (firstPart.ne.expectedFirstPart) then
         close(44)
         if(myrank.eq.0)write(69,*)"When ncooling=2 or 3, the "//
     $        "opacityfile must begin with '      logT,logR'"
         error stop "opacityfile problem. See log."
      end if
      close(44)
      
      open(44,file=trim(adjustl(opacityfile)),status='old')
      call fseek(44, len(firstPart), 0)
      read(44,'(A)') line
      close(44)
      
      ncols = 1
      do i=1, len(trim(adjustl(line)))-1
         if ( line(i:i) .eq. " " .and.
     $        line((i+1):(i+1)) .ne. " ") ncols = ncols + 1
      end do
      
 300  format("(",I0,"E15.7)")
      write(fmt,300) ncols
      write(fmt2,300) ncols+1
      
      open(44,file=trim(adjustl(opacityfile)),status='old')
      call fseek(44, len(firstPart), 0)
      read(44,trim(adjustl(fmt))) (logR(i), i=1, ncols)

      do i=1, size(logT)
         read(44,trim(adjustl(fmt2)),END=500) !dummy
     $        logT(i), (opacityTable(i,j),j=1,ncols)
      end do
      read(44,*,END=500)
      if(myrank.eq.0)write(69,*) "The logT and opacityTable arrays "//
     $     "are too small to hold the provided opacity file"
      error stop "arrays too small to hold opacityfile. See log."
 500  close(44)
      

      if(myrank.eq.0) write(*,*) "init_cooling completed successfully"





c     Use this code to test the getOpacity function, to see if it
c     recovers the same opacity file as was input initially
      if(debug.gt.0) then
         testRows = 100
         testCols = 100
         testlogRmin = minval(logR)
         testlogTmin = minval(logT)
         testdlogR = (maxval(logR)-testlogRmin)/dble(testCols-1)
         testdlogT = (maxval(logT)-testlogTmin)/dble(testRows-1)
         
         open(55,file=trim(adjustl(opacityfile))//"_test",
     $        status='unknown')

 880     format("(A",I0,",",I0,"E15.7)")
         write(fmt3,880) len(firstPart)-1, testCols
         
         write(55,fmt3) firstPart,
     $        (testlogRmin + testdlogR*(i-1),i=1,testCols)
         do i=1, testRows
            testlogT = testlogTmin + testdlogT*(i-1)
            testT = 10.d0**testlogT
            write(55,fmt2) testlogT, (getOpacity(
     $         10.d0**(testlogRmin+testdlogR*(j-1)+3.d0*testlogT-18.d0),
     $           testT),
     $           j=1,testCols)
         end do
         close(55)

         stop "Created test opacity file."
      end if

      return
      end
