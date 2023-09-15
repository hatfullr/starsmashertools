c     Similar to get_ncooling2, except now we also calculate the heating in
c     addition to the cooling. The total uraddot at the end is the cooling (a
c     negative quantity) plus the heating (a positive quantity).
      subroutine get_ncooling3
      use nearby_particles
      include '../starsmasher.h'
      include 'cooling.h'
      include 'mpif.h'
      integer i, mylength, ierr
      real start, finish
      
      if (myrank.eq.0) call cpu_time(start)
      
      call nearby_particles_init

c     Initialize arrays
      do i=1,n
         uraddot(i) = 0.d0
         uraddotcool(i) = 0.d0
         uraddotheat(i) = 0.d0
         myuraddotheat(i) = 0.d0
         dEemergdt(i) = 0.d0
         dEdiffdt(i) = 0.d0
         dEmaxdiffdt(i) = 0.d0
      end do

      if (cooling_type.eq.0) then ! fluffy
         do i=n_lower, n_upper
c     Do not let core particles cool at all
            if(u(i).ne.0) call fluffy(i)
         end do
      else if (cooling_type.eq.1) then ! dense
         do i=n_lower, n_upper
c     Do not let core particles cool at all
            if(u(i).ne.0) call dense(i)
         end do
      end if
      
      mylength=n_upper-n_lower+1
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     uraddotcool, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      call mpi_allreduce(myuraddotheat,uraddotheat,n,
     $     mpi_double_precision,mpi_sum,mpi_comm_world,ierr)
      
      do i=n_lower, n_upper
         uraddot(i) = uraddotcool(i) + uraddotheat(i)
      end do
      
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     uraddot, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     dEemergdt, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     dEdiffdt, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     dEmaxdiffdt, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      
      call nearby_particles_finalize
      
      updateduraddot=.true.

      if (myrank.eq.0) then
         call cpu_time(finish)
         write(*,'(a,3f10.3)') "cooling: ",finish-start,
     $        getnearby_time, getnearest_time
      end if
      
      return
      end subroutine
