      subroutine dense(i)
      use nearby_particles
c      use ifport
      include '../starsmasher.h'
      include 'cooling.h'
      integer i, iangle, nearest_j
      real*8 tdiff, theta, phi, sintheta,A,betai
      real*8 dEraddt, Uradi, Uradj, deltar
      integer js_to_heat(Nrays)
      integer ip, j, njs_to_heat
      real*8 gradUrads(Nrays)
      
c     The following are for efficiency
      real*8 xi,yi,zi,ri,ri2,factor
      
      real*8 rij
      real*8 qtau

      ri = rout(i)              ! set in kappa_and_T.f
      ri2 = ri*ri
      
      tdiff = popacity(i)*rho(i)*ri2 / crad_codeunits

      if (dt/tdiff.ge.1.d-12 .and. temperatures(i).gt.teq) then
         Uradi = arad_codeunits * temperatures(i)**4.d0

         A = 4.d0*pi*ri2
         dEemergdt(i) = 0.5d0*A*crad_codeunits*Uradi*qtau(tau(i))
         
         dEmaxdiffdt(i) = u(i)*am(i) / tdiff
         
c     This populates the array "nearby" with the indices of particles whose
c     kernel overlaps particle i's kernel anywhere.
         call nearby_particles_getnearby(i)
         njs_to_heat = 0
         do iangle=1, Nrays
            theta = icoangles(iangle,1)+(rand()-0.5d0)*thetawindow
            phi = icoangles(iangle,2)+(rand()-0.5d0)*phiwindow

            sintheta = sin(theta)
            call nearby_particles_get_nearest(
     $           xi + ri*sintheta*cos(phi),
     $           yi + ri*sintheta*sin(phi),
     $           zi + ri*cos(theta),
     $           nearest_j, rij
     $      )
            
            Uradj = 0.d0
            deltar = ri
            
            if (nearest_j.ne.-1) then ! Overlapping kernels exist
c     Don't cool if T_j >= T_i
               if (temperatures(nearest_j).ge.temperatures(i)) cycle
               Uradj = arad_codeunits*temperatures(nearest_j)**4.d0
               deltar = rij

               njs_to_heat = njs_to_heat + 1
               js_to_heat(njs_to_heat) = nearest_j
               gradUrads(njs_to_heat) = (Uradj - Uradi)/deltar
            end if
            
            dEdiffdt(i) = dEdiffdt(i) + (Uradj - Uradi)/deltar
         end do
         
         dEdiffdt(i) = -crad_codeunits/(3.d0*popacity(i)*rho(i)) * A *
     $        invNrays * dEdiffdt(i)

c     Nata: added a sanity check below
         betai = Uradi / (rho(i)*u(i))
         if (dEdiffdt(i)-(dEmaxdiffdt(i)*betai).gt.epsilon(1.d0)) then
            if(myrank.eq.0) then
               write(*,*) " Too high Ediffdt  ",dEdiffdt(i),
     $              dEmaxdiffdt(i)
               write(*,*) " check quantities  ", betai,
     $              dEmaxdiffdt(i)*betai, epsilon(1.d0)
            end if
            error stop "dense.f"
         end if


         uraddotcool(i)=-min(dEemergdt(i),dEdiffdt(i),
     $        dEmaxdiffdt(i)) / am(i)


         if (dEdiffdt(i).le.dEemergdt(i)) then
            factor = -crad_codeunits/(3.d0*popacity(i)*rho(i)) * A *
     $           invNrays
            do ip=1, njs_to_heat
               j = js_to_heat(ip)
               if (nearest_j.ne.-1) then ! Overlapping kernels exist
c     Send the radiation which we cooled particle i by to particle nearest_j as
c     a heating event
                  myuraddotheat(j)=myuraddotheat(j)+factor*gradUrads(ip)
     $                 * (1.d0 - exp(-tau(j))) / am(j)
               end if
            end do
         end if
         
      end if
      
      return
      end subroutine
