c     Create an icosahedron with some number of subdivisions
      subroutine init_icosahedron(subdiv)
      include "../starsmasher.h"
      include "cooling.h"
      integer subdiv
      
      integer i, j, k
      real*8 phi
c     Golden ratio
      parameter(phi=0.5d0*(1.d0+sqrt(5.d0)))
      real*8, dimension(:,:), allocatable :: verts
      integer, dimension(:,:), allocatable :: faces, faces_subdiv
      integer, dimension(:,:), allocatable :: already_cut, cut_temp
      integer, dimension(:), allocatable :: middle_point_cache, temp
      logical isAlreadyCut(3)
      integer notAlreadyCutSum
      integer minface12,minface23,minface31,maxface12,maxface23,
     $     maxface31
      integer already_cut_index, v1, v2, v3
      integer middle_point_index, faces_subdiv_index,p1,p2
      integer vertstoragenumber
      real*4 nan

      if(subdiv.gt.subdivmax) then
         write(*,*) "Cannot use more than ",subdivmax," subdivisions."//
     $        " Received ",subdiv," subdivisions"
         error stop "icosahedron_init"
      end if
      
c      write(*,*) "icosahedron_init started"
c      write(*,*) "# subdivisions = ",subdiv
c      write(*,*) "# vertices = ",10*(2**(subdiv))**2 + 2
      
      nan = 0.d0
      nan = 0.d0 / nan

c     Make sure all allocatable arrays have at least some allocation
      allocate(verts(10*(2**(subdiv))**2 + 2,3))
      allocate(faces(20,3)) 
      allocate(faces_subdiv(1,3))
      allocate(already_cut(1,2))
      allocate(cut_temp(1,2))
      allocate(middle_point_cache(1))
      allocate(temp(1))
      
      faces = transpose(reshape((/
c     5 faces around point 0
     $     1, 12,  6,
     $     1,  6,  2,
     $     1,  2,  8,
     $     1,  8, 11,
     $     1, 11, 12,
c     Adjacent faces
     $     2,  6, 10,
     $     6, 12,  5,
     $    12, 11,  3,
     $    11,  8,  7,
     $     8,  2,  9,
c     5 faces around 3
     $     4, 10,  5,
     $     4,  5,  3,
     $     4,  3,  7,
     $     4,  7,  9,
     $     4,  9, 10,
c     Adjacent faces
     $     5, 10,  6,
     $     3,  5, 12,
     $     7,  3, 11,
     $     9,  7,  8,
     $    10,  9,  2 /), (/3,20/)))

      

      
c     These are the base values for the vertices
      call icosahedron_vertex(-1.d0, phi, 0.d0,
     $     verts(1,1), verts(1,2), verts(1,3))
      call icosahedron_vertex( 1.d0, phi, 0.d0,
     $     verts(2,1), verts(2,2), verts(2,3))
      call icosahedron_vertex(-1.d0,-phi, 0.d0,
     $     verts(3,1), verts(3,2), verts(3,3))
      call icosahedron_vertex( 1.d0,-phi, 0.d0,
     $     verts(4,1), verts(4,2), verts(4,3))
      
      call icosahedron_vertex(0.d0,-1.d0, phi,
     $     verts(5,1), verts(5,2), verts(5,3))
      call icosahedron_vertex(0.d0, 1.d0, phi,
     $     verts(6,1), verts(6,2), verts(6,3))
      call icosahedron_vertex(0.d0,-1.d0,-phi,
     $     verts(7,1), verts(7,2), verts(7,3))
      call icosahedron_vertex(0.d0, 1.d0,-phi,
     $     verts(8,1), verts(8,2), verts(8,3))
      
      call icosahedron_vertex( phi,0.d0,-1.d0,
     $     verts( 9,1), verts( 9,2), verts( 9,3))
      call icosahedron_vertex( phi,0.d0, 1.d0,
     $     verts(10,1), verts(10,2), verts(10,3))
      call icosahedron_vertex(-phi,0.d0,-1.d0,
     $     verts(11,1), verts(11,2), verts(11,3))
      call icosahedron_vertex(-phi,0.d0, 1.d0,
     $     verts(12,1), verts(12,2), verts(12,3))

      vertstoragenumber = 12

c     We need to keep track of the vertices we have already cut so
c     that we do not accidentally cut them multiple times

      already_cut_index = 0
      middle_point_index = 0

c      write(*,*) "size = ",size(already_cut,1)
      do i=1, subdiv

c         open(13,file="faces.dat",status='unknown')
c         do j=1, size(faces,1)
c            write(13,'(3I3)') (faces(j,k), k=1, 3)
c         end do
c         close(13)

         deallocate(faces_subdiv)
         allocate(faces_subdiv(size(faces,1)*4,3))
         faces_subdiv_index = 1
         
         do j=1, size(faces,1)
            minface12 = min(faces(j,1),faces(j,2))
            maxface12 = max(faces(j,1),faces(j,2))
            minface23 = min(faces(j,2),faces(j,3))
            maxface23 = max(faces(j,2),faces(j,3))
            minface31 = min(faces(j,3),faces(j,1))
            maxface31 = max(faces(j,3),faces(j,1))

            isAlreadyCut(1) = .false.
            isAlreadyCut(2) = .false.
            isAlreadyCut(3) = .false.

            notAlreadyCutSum = 3
            
            do k=1, already_cut_index
               if ( .not.isAlreadyCut(1).and.
     $              already_cut(k,1).eq.minface12 .and.
     $              already_cut(k,2).eq.maxface12 ) then
                  isAlreadyCut(1) = .true.
                  v1 = middle_point_cache(k)
                  notAlreadyCutSum = notAlreadyCutSum - 1
               else if(.not.isAlreadyCut(2).and.
     $                 already_cut(k,1).eq.minface23 .and.
     $                 already_cut(k,2).eq.maxface23) then
                  isAlreadyCut(2) = .true.
                  v2 = middle_point_cache(k)
                  notAlreadyCutSum = notAlreadyCutSum - 1
               else if(.not.isAlreadyCut(3).and.
     $                 already_cut(k,1).eq.minface31 .and.
     $                 already_cut(k,2).eq.maxface31) then
                  isAlreadyCut(3) = .true.
                  v3 = middle_point_cache(k)
                  notAlreadyCutSum = notAlreadyCutSum - 1
               end if
            end do

c     Make room in the already_cut array and the middle_point_cache array
            if ( .not.all(isAlreadyCut) ) then
               deallocate(cut_temp)
               allocate(cut_temp(already_cut_index,2))
               do k=1, already_cut_index
                  cut_temp(k,1) = already_cut(k,1)
                  cut_temp(k,2) = already_cut(k,2)
               end do
               deallocate(already_cut)
               allocate(
     $              already_cut(already_cut_index+notAlreadyCutSum,2))
               do k=1, already_cut_index
                  already_cut(k,1) = cut_temp(k,1)
                  already_cut(k,2) = cut_temp(k,2)
               end do

               deallocate(temp)
               allocate(temp(size(middle_point_cache)))
               do k=1, size(temp)
                  temp(k) = middle_point_cache(k)
               end do
               deallocate(middle_point_cache)
               allocate(middle_point_cache(
     $              size(temp)+notAlreadyCutSum))
               do k=1, size(temp)
                  middle_point_cache(k) = temp(k)
               end do
            end if

            if ( .not.isAlreadyCut(1) ) then
               vertstoragenumber = vertstoragenumber + 1
               if ( vertstoragenumber .gt. size(verts,1) ) goto 500

               p1 = faces(j,1)
               p2 = faces(j,2)
               call icosahedron_vertex_middle(
     $              verts(p1,1),verts(p1,2),verts(p1,3),
     $              verts(p2,1),verts(p2,2),verts(p2,3),
     $              verts(vertstoragenumber,1),
     $              verts(vertstoragenumber,2),
     $              verts(vertstoragenumber,3))
               v1 = vertstoragenumber
               
c     Append to the already_cut array
               already_cut_index = already_cut_index + 1
               already_cut(already_cut_index,1)=minface12
               already_cut(already_cut_index,2)=maxface12

c     Append to the middle_point_cache array
               middle_point_index = middle_point_index + 1
               middle_point_cache(middle_point_index)=v1
            end if

            if ( .not.isAlreadyCut(2) ) then
               vertstoragenumber = vertstoragenumber + 1
               if ( vertstoragenumber .gt. size(verts,1) ) goto 500
               p1 = faces(j,2)
               p2 = faces(j,3)
               call icosahedron_vertex_middle(
     $              verts(p1,1),verts(p1,2),verts(p1,3),
     $              verts(p2,1),verts(p2,2),verts(p2,3),
     $              verts(vertstoragenumber,1),
     $              verts(vertstoragenumber,2),
     $              verts(vertstoragenumber,3))
               v2 = vertstoragenumber
               
c     Append to the already_cut array
               already_cut_index = already_cut_index + 1
               already_cut(already_cut_index,1)=minface23
               already_cut(already_cut_index,2)=maxface23

c     Append to the middle_point_cache array
               middle_point_index = middle_point_index + 1
               middle_point_cache(middle_point_index)=v2
            end if

            if ( .not.isAlreadyCut(3) ) then
               vertstoragenumber = vertstoragenumber + 1
               if ( vertstoragenumber .gt. size(verts,1) ) goto 500
               p1 = faces(j,3)
               p2 = faces(j,1)
               call icosahedron_vertex_middle(
     $              verts(p1,1),verts(p1,2),verts(p1,3),
     $              verts(p2,1),verts(p2,2),verts(p2,3),
     $              verts(vertstoragenumber,1),
     $              verts(vertstoragenumber,2),
     $              verts(vertstoragenumber,3))
               v3 = vertstoragenumber
               
c     Append to the already_cut array
               already_cut_index = already_cut_index + 1
               already_cut(already_cut_index,1)=minface31
               already_cut(already_cut_index,2)=maxface31

c     Append to the middle_point_cache array
               middle_point_index = middle_point_index + 1
               middle_point_cache(middle_point_index)=v3
            end if
            
c     Append to the faces_subdiv array
            faces_subdiv(faces_subdiv_index,1) = faces(j,1)
            faces_subdiv(faces_subdiv_index,2) = v1
            faces_subdiv(faces_subdiv_index,3) = v3
            faces_subdiv_index = faces_subdiv_index + 1

            faces_subdiv(faces_subdiv_index,1) = faces(j,2)
            faces_subdiv(faces_subdiv_index,2) = v2
            faces_subdiv(faces_subdiv_index,3) = v1
            faces_subdiv_index = faces_subdiv_index + 1

            faces_subdiv(faces_subdiv_index,1) = faces(j,3)
            faces_subdiv(faces_subdiv_index,2) = v3
            faces_subdiv(faces_subdiv_index,3) = v2
            faces_subdiv_index = faces_subdiv_index + 1

            faces_subdiv(faces_subdiv_index,1) = v1
            faces_subdiv(faces_subdiv_index,2) = v2
            faces_subdiv(faces_subdiv_index,3) = v3
            faces_subdiv_index = faces_subdiv_index + 1            
            
         end do

c     Replace the faces array with the content of the faces_subdiv array
         deallocate(faces)
         allocate(faces(size(faces_subdiv,1),size(faces_subdiv,2)))
         do j=1, size(faces,1)
            do k=1, size(faces,2)
               faces(j,k) = faces_subdiv(j,k)
            end do
         end do
      end do

c     Initialize the icoangles array with nan values that hopefully
c     cause obvious problems when accidentally accessed
      do i=1, size(icoangles,1)
         icoangles(i,1) = nan
         icoangles(i,2) = nan
      end do
      

c     Calculate the angles of each vertex
c      open(14, file="angles.dat", status='unknown')
      do i=1, size(verts,1)
         icoangles(i,1) = acos(verts(i,3)) ! theta
         icoangles(i,2) = atan2(verts(i,2),verts(i,1)) ! phi
c         write(14,'(2E15.7)') icoangles(i,1),icoangles(i,2)
      end do
c      close(14)

      
      
 500  continue

      
c     To debug, write all the vertices to a file
c      open(12, file="vertices.dat", status="unknown")
c      do i=1, size(verts,1)
c         write(12,'(3E15.7)') verts(i,1), verts(i,2), verts(i,3)
c      end do
c      close(12)

c     Start cleanup
      if(myrank.eq.0) then
         write(*,*) "icosahedron_init cleaning up. If you see an "//
     $        "error 'free(): invalid pointer' here "//
     $        "then it is a problem with the icosahedron code. If "//
     $        "that happens, try messing with the subdiv and "//
     $        "maxsubdiv variables first."
      end if
      if (allocated(faces)) deallocate(faces)
      if (allocated(faces_subdiv)) deallocate(faces_subdiv)
      if (allocated(already_cut)) deallocate(already_cut)
      if (allocated(cut_temp)) deallocate(cut_temp)
      if (allocated(middle_point_cache)) deallocate(middle_point_cache)
      if (allocated(temp)) deallocate(temp)
      
c     Catch errors
      if ( vertstoragenumber .gt. size(verts,1) ) then
         write(*,*) "vertstoragenumber = ",vertstoragenumber
         write(*,*) "size(verts,1) = ",size(verts,1)
         write(*,*) "last element in verts = ",
     $        verts(size(verts,1)-1,1),
     $        verts(size(verts,1)-1,2),
     $        verts(size(verts,1)-1,3)
         deallocate(verts)
         if(myrank.eq.0)write(69,*)"Tried to store more vertices "//
     $        "than allowed for this icosahedron"
         error stop "Too many vertices in icosahedron. See log."
      end if

c     Finish cleanup
      if (allocated(verts)) deallocate(verts)

c      write(*,*) "icosahedron_init completed successfully"

c     Record the total number of rays
      Nrays = 10*(2**subdiv)**2+2
      invNrays = 1.d0/float(Nrays)
      
      if(myrank.eq.0) then
         write(69,*) "Using icosahedron angles:"
         do i=1,Nrays
            write(69,*) icoangles(i,1), icoangles(i,2)
         end do
c         call flush(69)
      end if
      
      return
      
      end
      

c     Return vertex coordinates fixed to the unit sphere
      subroutine icosahedron_vertex(x,y,z,xvert,yvert,zvert)
      implicit none
      real*8 x,y,z,xvert,yvert,zvert,length
      length = sqrt(x*x + y*y + z*z)
      xvert = x / length
      yvert = y / length
      zvert = z / length
      return
      end

c     Find the middle point and project it to the unit sphere
      subroutine icosahedron_vertex_middle(x1,y1,z1,x2,y2,z2,x,y,z)
      implicit none
      real*8 x1,y1,z1,x2,y2,z2,x,y,z
      call icosahedron_vertex(0.5d0*(x1+x2),0.5d0*(y1+y2),0.5d0*(z1+z2),
     $     x,y,z)
      return
      end
