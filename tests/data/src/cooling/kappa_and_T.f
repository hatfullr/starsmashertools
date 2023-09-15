c     Calculate the opacities and temperatures of all particles

      subroutine kappa_and_T
      include '../starsmasher.h'
      include 'cooling.h'
      include 'mpif.h'
      integer i
      real*8 rhocgs, ucgs, uunit, rhounit, opacunit
      real*8 getOpacity, useeostable
      integer mylength, irank, ierr
      real start, finish
c      real*8 taucoef
c      common/tau/taucoef

      if(myrank.eq.0) then
         write(69,*) "Obtaining opacities and temperatures in "//
     $        "subroutine kappa_and_T using neos=",neos
         call flush(69)
         call cpu_time(start)
      end if

c     Just in case, synchronize relavent quantities
      mylength=n_upper-n_lower+1
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     rho, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     u, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     meanmolecular, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     am, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     hp, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)

      
      uunit=gravconst*munit/runit
      rhounit=munit/runit**3
      opacunit=runit*runit/munit

      if (neos.eq.1) then
         do i=n_lower, n_upper
            if ( u(i).ne.0.d0 ) then
               rhocgs = rho(i)*rhounit
               if(nintvar.eq.1) then
                  ucgs=u(i)*rho(i)**(gam-1.d0)/(gam-1.d0)*uunit
               else
                  ucgs=u(i)*uunit
               endif
               call gettemperature(qconst*rhocgs/meanmolecular(i),
     $              -ucgs*rhocgs/arad,temperatures(i))
c     The opacity file contains some -1.d30 values which are used to
c     indicate an error. We glide over those errors by instead making
c     the particle unrealistically transparent
               popacity(i) = max(getOpacity(rhocgs,temperatures(i)),
     $              1.d-30)/opacunit
            else
               temperatures(i) = 0.d0
               popacity(i) = 0.d0
            end if
         end do
      else if (neos.eq.2) then
         do i=n_lower, n_upper
            if ( u(i).ne.0.d0 ) then
               rhocgs = rho(i)*rhounit
               if(nintvar.eq.1) then
                  ucgs=u(i)*rho(i)**(gam-1.d0)/(gam-1.d0)*uunit
               else
                  ucgs=u(i)*uunit
               endif
               temperatures(i) = useeostable(ucgs,rhocgs,
     $              meanmolecular(i),1)
c     The opacity file contains some -1.d30 values which are used to
c     indicate an error. We glide over those errors by instead making
c     the particle unrealistically transparent
               popacity(i) = max(getOpacity(rhocgs,temperatures(i)),
     $              1.d-30)/opacunit
            else
               temperatures(i) = 0.d0
               popacity(i) = 0.d0
            end if
         end do
      else
         stop 'radiative cooling requires neos=1 or 2'
      endif

c     Calculate the particle individual optical depths and rout values
      if (cooling_type.eq.0) then ! fluffy
         do i=n_lower, n_upper
            rout(i) = 2.d0*hp(i)
         end do
      else if (cooling_type.eq.1) then ! dense
         do i=n_lower, n_upper
            rout(i) = (0.75d0*am(i)/(pi*rho(i)))**(0.3333333333333333d0)
         end do
      else if (myrank.eq.0) then
         write(*,*) "Input parameter 'cooling_type' must be one of"//
     $        " 0 for 'fluffy' or 1 for 'dense'"
         error stop "kappa_and_T.f"
      end if
      
      do i=n_lower, n_upper
c     tau(i) = taucoef * am(i) * popacity(i) / (hp(i)*hp(i))
         tau(i) = popacity(i) * rho(i) * rout(i)
      end do

c     Synchronize the threads
      mylength=n_upper-n_lower+1
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     popacity, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     temperatures, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     tau, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)
      call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision,
     $     rout, recvcounts,displs, mpi_double_precision,
     $     mpi_comm_world, ierr)

      if (myrank.eq.0) then
         call cpu_time(finish)
         write(*,'(a,f10.3,a,i4)') "kappa_and_T: ", finish-start
      end if
      
c      if(myrank.eq.0) then
c         write(69,*) "kappa_and_T finished"
c         call flush(69)
c      end if
      
      end subroutine
