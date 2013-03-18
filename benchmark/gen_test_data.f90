      program gen_test_data

        implicit none

        integer, parameter :: nb=500
        integer :: i
        double precision :: p=0.1, db
        double precision, dimension(nb) :: b

        double precision :: u1=0.3, u2=0.1
        double precision, dimension(nb) :: muo1, mu0

        db = (1.d0 + p) / dble(nb - 1)
        do i=1,nb
          b(i) = dble(i - 1) * db
        enddo

        call occultquad(b, u1, u2, p, muo1, mu0, nb)

        write(*,*) "# ", u1, u2
        do i=1,nb
          write(*,*) b(i), muo1(i)
        enddo

      end program
