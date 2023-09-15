      function getOpacity(rhocgs,temperature)
      use bilinear_interpolation
c      include '../starsmasher.h'
      real*8 getOpacity,rhocgs,temperature
      real*8 logR(1000), logT(1000)
      common/lrt/ logR, logT
      real*8 opacityTable(1000,1000)
      common/opacitytablecomm/opacityTable
      real*8 mylogR, mylogT
      
      if(temperature.lt.3.d3)then
         getOpacity = 1.d-4
      else
         mylogT = log10(temperature)
         mylogR = log10(rhocgs) - 3.d0*mylogT + 18.d0
      
         getOpacity = bilinear_interpolate(
     $        size(logT),logT,
     $        size(logR),logR,
     $        opacityTable,
     $        mylogT, mylogR)
      endif
      
      return
      end
