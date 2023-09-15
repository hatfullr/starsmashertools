c     Icosahedron variables
      integer subdivmax
      parameter(subdivmax=5)
      real*8 icoangles(10*(2**(subdivmax))**2+2,2)
      common/icoanglescomm/icoangles
      
      real*8 crad_codeunits, arad_codeunits, sigma_codeunits
      common/codeunits/ crad_codeunits,arad_codeunits,sigma_codeunits

      
      integer Nrays, cooling_type,
c     This is an input variable. See namelist in init.f
     $     icosahedron_subdivisions
      common/coolingints/ Nrays,cooling_type,icosahedron_subdivisions

      real*8 invNrays, tcool, cooling_update_frequency,
c     These are input variables. See namelist in init.f      
     $     max_cooling_update_frequency, thetawindow, phiwindow
      common/coolingreals/max_cooling_update_frequency,tcool,
     $     cooling_update_frequency,invNrays,thetawindow,phiwindow

      logical updateduraddot
      common/coolingbools/ updateduraddot
