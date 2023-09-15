c     From https://github.com/cfinch/Shocksolution_Examples/blob/master/FORTRAN/BilinearInterpolation/interpolation.f90
      module bilinear_interpolation

      contains

      function binarysearch(length, array, value, delta)
      ! Given an array and a value, returns the index of the element that
      ! is closest to, but less than, the given value.
      ! Uses a binary search algorithm.
      ! "delta" is the tolerance used to determine if two values are equal
      ! if ( abs(x1 - x2) <= delta) then
      !    assume x1 = x2
      ! endif

      implicit none
      integer, intent(in) :: length
      real*8, dimension(length), intent(in) :: array
      !f2py depend(length) array
      real*8, intent(in) :: value
      real*8, intent(in), optional :: delta
      
      integer :: binarysearch
      
      integer :: left, middle, right
      real*8 :: d
      
      if (present(delta) .eqv. .true.) then
         d = delta
      else
         d = 1e-9
      endif
      
      left = 1
      right = length
      do
         if (left > right) then
            exit
         endif
         middle = nint((left+right) / 2.0)
         if ( abs(array(middle) - value) <= d) then
            binarySearch = middle
            return
         else if (array(middle) > value) then
            right = middle - 1
         else
            left = middle + 1
         end if
      end do
      binarysearch = right
      
      end function binarysearch
      
      real*8 function bilinear_interpolate(x_len, x_array, y_len,
     $     y_array,f, x, y, delta)
      ! This function uses bilinear interpolation to estimate the value
      ! of a function f at point (x,y)
      ! f is assumed to be sampled on a regular grid, with the grid x values specified
      ! by x_array and the grid y values specified by y_array
      ! Reference: http://en.wikipedia.org/wiki/Bilinear_interpolation
      implicit none
      integer, intent(in) :: x_len, y_len           
      real*8, dimension(x_len), intent(in) :: x_array
      real*8, dimension(y_len), intent(in) :: y_array
      real*8, dimension(x_len, y_len), intent(in) :: f
      real*8, intent(in) :: x,y
      real*8, intent(in), optional :: delta
      !f2py depend(x_len) x_array, f
      !f2py depend(y_len) y_array, f
      
      real*8 :: denom, x1, x2, y1, y2
      integer :: i,j,i1, i2, j1, j2
      
      i = binarysearch(x_len, x_array, x)
      j = binarysearch(y_len, y_array, y)

      i2 = min(x_len, i+1)
      i1 = i2-1
      x1 = x_array(i1)
      x2 = x_array(i2)

      j2 = min(y_len, j+1)
      j1 = j2-1
      y1 = y_array(j1)
      y2 = y_array(j2)
      
      denom = (x2 - x1)*(y2 - y1)
      
      bilinear_interpolate = (f(i1,j1)*(x2-x)*(y2-y) +
     $     f(i2,j1)*(x-x1)*(y2-y) +f(i1,j2)*(x2-x)*(y-y1) +
     $     f(i2, j2)*(x-x1)*(y-y1))/denom
      
      end function bilinear_interpolate









      SUBROUTINE FIND2D(Xwant,Ywant,Value,X,Y,Array,NX,NY)
      IMPLICIT NONE
!     
!     Dummy arguments
!     
      INTEGER :: NX , NY
      REAL*8 :: Value , Xwant , Ywant
      REAL*8 , DIMENSION(NX,NY) :: Array
      REAL*8 , DIMENSION(NX) :: X
      REAL*8 , DIMENSION(NY) :: Y
      INTENT (IN) NX , NY
!
!     Local variables
!
      INTEGER :: jx , jxp1 , jy , jyp1
!
!-----------------------------------------------------------------------
!
!     written by:   David Lary
!
!     started:      7/1/1993
!
!     last updated: 22/1/2004
!
!----------------------------------------------------------------------
!
!     2D bilinear interpolation.
!
!-----------------------------------------------------------------------
!
      CALL FINDINDEX(Xwant,Ywant,X,Y,NX,NY,jx,jxp1,jy,jyp1)
      CALL FINDPOINT(Xwant,Ywant,Value,X,Y,Array,NX,NY,jx,jxp1,jy,jyp1)
!
!-----------------------------------------------------------------------
!
      END SUBROUTINE FIND2D
      
      

      SUBROUTINE FINDINDEX(Xwant,Ywant,X,Y,NX,NY,Jx,Jxp1,Jy,Jyp1)
      IMPLICIT NONE
!
!
! Dummy arguments
!
      INTEGER :: Jx , Jxp1 , Jy , Jyp1
      INTEGER :: NX , NY
      REAL*8 :: Xwant , Ywant
      REAL*8 , DIMENSION(NX) :: X
      REAL*8 , DIMENSION(NY) :: Y
      INTENT (IN) NX , NY
      INTENT (OUT) Jxp1 , Jyp1
      INTENT (INOUT) Jx , Jy
!     
!-----------------------------------------------------------------------
!
!     written by:   David Lary
!
!     started:      7/1/1993
!     
!     last updated: 22/1/2004
!
!----------------------------------------------------------------------
!
!     Find grid references.
!     
!-----------------------------------------------------------------------
!
!     i) x
      CALL POS(X,NX,Xwant,Jx)
      IF ( Jx==0 ) THEN
         Jx = 1
      ELSEIF ( Jx>=NX ) THEN
         Jx = NX - 1
      ENDIF
      Jxp1 = Jx + 1
!     
!     ii) y
      CALL POS(Y,NY,Ywant,Jy)
      IF ( Jy==0 ) THEN
         Jy = 1
      ELSEIF ( Jy>=NY ) THEN
         Jy = NY - 1
      ENDIF
      Jyp1 = Jy + 1
!
!-----------------------------------------------------------------------
!
      END SUBROUTINE FINDINDEX
      
      


      SUBROUTINE FINDPOINT(Xwant,Ywant,Value,X,Y,Array,NX,NY,Jx,Jxp1,Jy,
     $     Jyp1)
      IMPLICIT NONE
!
!
! Dummy arguments
!
      INTEGER :: Jx , Jxp1 , Jy , Jyp1
      INTEGER :: NX , NY
      REAL*8 :: Value , Xwant , Ywant
      REAL*8 , DIMENSION(NX,NY) :: Array
      REAL*8 , DIMENSION(NX) :: X
      REAL*8 , DIMENSION(NY) :: Y
      INTENT (IN) Array , Jx , Jxp1 , Jy , Jyp1 , NX , NY , X , Xwant , 
     $     Y , Ywant
      INTENT (OUT) Value
!
! Local variables
!
      REAL*8 :: s1 , s2 , s3 , s4 , zt , zta , ztb , zu , zua , zub
!
!-----------------------------------------------------------------------
!
!     written by:   David Lary
!
!     started:      7/1/1993
!
!     last updated: 22/1/2004
!
!-----------------------------------------------------------------------
!
!     look up relevant grid points.
!
!-----------------------------------------------------------------------
!
      s1 = Array(Jx,Jy)
      s2 = Array(Jxp1,Jy)
      s3 = Array(Jxp1,Jyp1)
      s4 = Array(Jx,Jyp1)
!
!     find slopes used in interpolation;
!     i) x.
      zta = Xwant - X(Jx)
      ztb = X(Jxp1) - X(Jx)
!
      zt = zta/ztb
!
!     ii) y.
      zua = Ywant - Y(Jy)
      zub = Y(Jyp1) - Y(Jy)
!
      zu = zua/zub
!
!     use bilinear interpolation.
      Value = (1.0-zt)*(1.0-zu)*s1 + zt*(1.0-zu)*s2 + zt*zu*s3 +        
     $     (1.0-zt)*zu*s4
!
!-----------------------------------------------------------------------
!
      END SUBROUTINE FINDPOINT


      SUBROUTINE POS(Xx,N,X,J)
      IMPLICIT NONE
!
! Dummy arguments
!
      INTEGER :: J
      INTEGER :: N
      REAL*8 :: X
      REAL*8 , DIMENSION(N) :: Xx
      INTENT (IN) N , X , Xx
      INTENT (OUT) J
!
! Local variables
!
      INTEGER :: jl , jm , ju
!
!-----------------------------------------------------------------------
!
!     Argrument list.
!
!     Name      Type              Description.
!     XX        Array of real     Monotonic array of length N.
!     (Unchanged on exit).
!
!     N         Integer           Length of array XX.
!     (Unchanged on exit).
!
!     X         Real              Value whose position in XX is
!     required.
!     (Unchanged on exit).
!
!     J         Integer           Index of X in array XX.
!     (Contains answer on exit).
!
!-----------------------------------------------------------------------
!
!     Given an array XX of length N, and given a value X, POS returns a
!     value J sucha that X lies between XX(J) and XX(J+1). XX must be
!     monotonic, either increasing or decreasing. J=0 or J=N is returned
!     to indicate that X is out of range.
!
!     The table entry J is found by bisection.
!
!     Based on Numerical Recipes, The art of scientific computing,
!     section 3.4, by Press, Flannery, Teukolsky & Vetterling,
!     Cambridge University Press, 1987.
!
!     Modified by: David Lary
!     ----------
!
!     Date Started : 7/2/1990
!
!     Last modified: 27/9/1991
!
!-----------------------------------------------------------------------
!
!     Initialize upper & lower limits.
      jl = 0
      ju = N + 1
!
!----------------------------------------------------------------------
!
      DO WHILE ( .TRUE. )
!
!----------------------------------------------------------------------
!
         IF ( .NOT..TRUE. ) THEN
            RETURN
!
         ELSEIF ( ju-jl>1 ) THEN
!
            jm = (ju+jl)/2
            IF ( Xx(N)>Xx(1) .EQV. X>Xx(jm) ) THEN
               jl = jm
            ELSE
               ju = jm
            ENDIF
!
!           Repeat untill the test condition is satisfied.
            CYCLE
         ENDIF
!
!        Set the output.
         J = jl
         EXIT
!
!----------------------------------------------------------------------
!
      ENDDO
!
!----------------------------------------------------------------------
!
      END SUBROUTINE POS








      

      SUBROUTINE polint(xa,ya,n,x,y,dy)
c     From Numerical Recipes in Fortran 77, 2nd Edition by Press, Teukolsky, Vetterling,
c     and Flannery (1992). Volume 1. Pages 103 and 104. Changed REAL to REAL*8. Removed
c     ridiculous do line identifier numbers to allow compilation.
      implicit none
      integer, intent(in) :: n
      integer :: NMAX,i,m,ns
      real*8, intent(inout) :: y, dy
      real*8, intent(in) :: x
      real*8, dimension(n), intent(in) :: xa,ya
      real*8 :: den, dif, dift, ho, hp, w

c      INTEGER n,NMAX
c      REAL*8 dy,x,y,xa(n),ya(n)
      PARAMETER (NMAX=10)       ! Largest anticipated value of n.
      real*8, dimension(NMAX) :: c,d
c     Given arrays xa and ya, each of length n, and given a value x, this routine returns
c     a value y, and an error estimate dy. If P(x) is the polynomial of degree N−1 such that
c     P(xa_i)=ya_i, i=1,...,n, then the returned value y = P(x).
c      INTEGER i,m,ns
c      REAL*8 den,dif,dift,ho,hp,w,c(NMAX),d(NMAX)
c      write(o,*) "Calling polint"
      ns=1
c      write(o,*) xa(1), dif
c      write(o,*) "Before dif"
      dif=abs(x-xa(1))
c      write(o,*) "After dif"
      do i=1,n                  ! Here we find the index ns of the closest table entry.
         dift=abs(x-xa(i))
         if (dift.lt.dif) then
            ns=i
            dif=dift
         endif
         c(i)=ya(i)             ! and initialize the tableau of c's and d's
         d(i)=ya(i)
      enddo
      y=ya(ns)                  ! This is the initial approximation to y.
      ns=ns-1
      do m=1,n-1                ! For each column ofthe tableau,
         do i=1,n-m             ! we loop over the current c's and d's and update them.
            ho=xa(i)-x
            hp=xa(i+m)-x
            w=c(i+1)-d(i)
            den=ho-hp
            if(den.eq.0.) error stop 'polint.f: failure in polint'
            ! This error can occur only if two input xa's are (to within roundoff) identical.
            den=w/den
            d(i)=hp*den         ! Here the c's and d's are updated.
            c(i)=ho*den
         enddo
         if (2*ns.lt.n-m)then   ! After each column in the tableau is completed, we decide
            dy=c(ns+1)          ! which correction, c or d, we want to add to our accu-
         else                   ! mulating value of y. i.e., which path to take through
            dy=d(ns)            ! the tableau--forking up or down. We do this in such a
            ns=ns-1             ! way as to take the most "straight line" route through the
         end if                 ! tableau to its apex, updating ns accordingly to keep track
         y=y+dy                 ! of where we are. This route keeps the partial approxima-
      enddo                     ! tions centered (insofar as possible) on the target x. The
      return                    ! last dy added is thus the error indication.
      END

      
      SUBROUTINE polin2(x1a,x2a,ya,m,n,x1,x2,y,dy)
c     From Numerical Recipes in Fortran 77, 2nd Edition by Press, Teukolsky, Vetterling,
c     and Flannery (1992). Volume 1. Page 118. Changed REAL to REAL*8. Removed ridiculous
c     do line identifier numbers to allow compilation.
      implicit none
      integer, intent(in) :: m,n
      integer :: NMAX,MMAX
      real*8, dimension(m), intent(in) :: x1a
      real*8, dimension(n), intent(in) :: x2a
      real*8, dimension(m,n), intent(in) :: ya
      real*8, intent(in) :: x1,x2
      real*8, intent(out) :: y,dy
c      INTEGER m,n,NMAX,MMAX
c      REAL*8 dy,x1,x2,y,x1a(m),x2a(n),ya(m,n)
      PARAMETER (NMAX=20,MMAX=20)                ! Maximum expected values of n and m
C     USES polint
c     Given arrays x1a(1:m) and x2a(1:n) of independent variables, and an m by n array of
c     function values ya(1:m,1:n), tabulated at the grid points defined by x1a and x2a; and
c     given values x1 and x2 of the independent variables; this routine returns an
c     interpolated function value y, and an accuracy indication dy (based only on the
c     interpolation in the x1 direction, however).
      integer :: j,k
      real*8, dimension(MMAX) :: ymtmp
      real*8, dimension(NMAX) :: yntmp
c      INTEGER j,k
c      REAL*8 ymtmp(MMAX), yntmp(NMAX)
c      write(o,*) "Calling polin2"
      do j=1,m                                   ! Loop over rows.
         do k=1,n                                ! Copy the row into temporary storage.
            yntmp(k)=ya(j,k)
         enddo
c         write(o,*) "End of k loop, j = ",j
c         write(o,*) "x2a = ",x2a
c         write(o,*) "yntmp = ",yntmp
c         write(o,*) "n = ",n
c         write(o,*) "x2 = ",x2
c         write(o,*) "ymtmp(j) = ",ymtmp(j)
c         write(o,*) "dy = ",dy

         call polint(x2a,yntmp,n,x2,ymtmp(j),dy) ! Interpolate answer into temporary stor-
      enddo                                      !     age.
      call polint(x1a,ymtmp,m,x1,y,dy)           ! Do the final interpolation.
      return
      END


      
      end module bilinear_interpolation
      
