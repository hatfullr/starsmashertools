      module nearby_particles
      implicit none

      integer, allocatable, dimension(:) :: nearby
      integer n_nearby

      real getnearby_time, getnearest_time

      contains

      subroutine nearby_particles_init
      include '../starsmasher.h'
      include 'cooling.h'
      allocate(nearby(n))
      if (myrank.eq.0) then
         getnearby_time = 0.d0
         getnearest_time = 0.d0
      end if
      end subroutine

      subroutine nearby_particles_finalize
      implicit none
      deallocate(nearby)
      end subroutine


c     Find all the particles that have their kernel overlapping particle i's
c     kernel anywhere.
      subroutine nearby_particles_getnearby(i)
      include '../starsmasher.h'
      include 'cooling.h'
      integer i, j
      real*8 xi,yi,zi, ri, dx,dy,dz,dr2,rsum
      real start, finish

      if (myrank.eq.0) call cpu_time(start)
      
      xi = x(i)
      yi = y(i)
      zi = z(i)
      ri = rout(i)

      n_nearby = 0
      
      do j=1, n
         if (i.ne.j .and. u(j).ne.0.d0) then
            dx = xi - x(j)      ! destination - origin
            dy = yi - y(j)
            dz = zi - z(j)
            dr2 = dx*dx + dy*dy + dz*dz
            rsum = rout(j) + ri
            if (dr2 .le. rsum*rsum) then
               n_nearby = n_nearby + 1
               nearby(n_nearby) = j
            end if
         end if
      end do

      if (myrank.eq.0) then
         call cpu_time(finish)
         getnearby_time = getnearby_time + (finish-start)
      end if
      
      end subroutine


c     Find the particle that is closest to the given position (excluding the
c     root particle) and obtain the gradient of the radiation energy density
c     Urad = aT^4, gradU = (Uradj - Uradi) / r_{ij}. Only particles whose
c     kernel encapsulates (xpos,ypos,zpos) can be chosen.
      subroutine nearby_particles_get_nearest(xpos,ypos,zpos,
     $     nearest_j,rij)
      include '../starsmasher.h'
      real*8, intent(in) :: xpos,ypos,zpos
      integer, intent(out) :: nearest_j
      real*8, intent(out) :: rij
      integer j, i
      real*8 rij2, dr2, dx, dy, dz
      real start, finish

      if (myrank.eq.0) call cpu_time(start)

c     A value of -1 indicates that no particles have a kernel which encapsulates
c     (xpos,ypos,zpos)
      nearest_j = -1
      rij2 = 1.d30
      do i=1, n_nearby
         j = nearby(i)
         
         dx = xpos-x(j)         ! destination - origin
         dy = ypos-y(j)
         dz = zpos-z(j)
         dr2 = dx*dx + dy*dy + dz*dz
         
         if (dr2.lt.rout(j)*rout(j)) then
            if (dr2.lt.rij2) then
               rij2 = dr2
               nearest_j = j
            end if
         end if
      end do

      rij = sqrt(rij2)

      if (myrank.eq.0) then
         call cpu_time(finish)
         getnearest_time = getnearest_time + (finish - start)
      end if
      
      return
      end subroutine
      
      end module


