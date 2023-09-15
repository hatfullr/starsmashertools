      subroutine fluffy(i)
      use nearby_particles
c      use ifport
      include '../starsmasher.h'
      include 'cooling.h'
      integer i, iangle, nearest_j
      real*8 tdiff, theta, phi, sintheta,A,betai
      real*8 dEraddt, Uradi, factor
      integer js_to_heat(Nrays)
      integer njs_to_heat, ip, j
      
c     The following are for efficiency
      real*8 xi,yi,zi,ri,ri2
      
      real*8 rij
      real*8 qtau
      
      ri = rout(i)              ! set in kappa_and_T.f
      ri2 = ri*ri
      
      tdiff = popacity(i)*rho(i)*ri2 / crad_codeunits
      
      if (dt/tdiff.ge.1.d-12 .and. temperatures(i).gt.teq) then
c     Get the cooling
         betai = arad_codeunits * temperatures(i)**4.d0 / (rho(i)*u(i))
c Nata: added 4 lines below
         if (betai.gt.1 .or. betai.le.0) then
            if(myrank.eq.0) then
               write(*,*) "beta is out of allowed range  ",betai
            end if
            error stop "fluffy.f"
         end if
         
         Uradi = betai*u(i)*am(i)*0.75d0/(pi*ri2*ri)
         
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
            
            if (nearest_j.ne.-1) then ! Overlapping kernels exist
c     Ignore this ray if T_j >= T_i. Assume that the heating of particle i will
c     be handled by the rays of other particles. We heat up particle j in a
c     separate loop below.
               if (temperatures(nearest_j).ge.temperatures(i)) cycle
               njs_to_heat = njs_to_heat+1
               js_to_heat(njs_to_heat) = nearest_j
            end if
            dEdiffdt(i) = dEdiffdt(i) + 1
         end do
         
         dEdiffdt(i) = betai * dEmaxdiffdt(i) * invNrays * dEdiffdt(i)
         uraddotcool(i)=-min(dEemergdt(i),dEdiffdt(i),dEmaxdiffdt(i))
     $        / am(i)



         
c     Get the heating
         if (dEdiffdt(i).le.dEemergdt(i)) then
            factor = betai*dEmaxdiffdt(i)*invNrays
            do ip=1, njs_to_heat
               j = js_to_heat(ip)
c     Send the radiation which we cooled particle i by to particle nearest_j as
c     a heating event
               myuraddotheat(j) = myuraddotheat(j) + factor *
     $              (1.d0 - exp(-tau(j))) / am(j)
            end do
         end if
      end if
      
      return
      end subroutine
