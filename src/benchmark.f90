      subroutine quadlaw (u1, u2, nr, r, ir)

        implicit none

        double precision, intent(in) :: u1, u2
        integer, intent(in) :: nr
        double precision, dimension(nr), intent(out) :: r, ir

        integer :: i
        double precision :: mu, dr

        dr = 1.0d0 / nr

        do i=1,nr

          r(i) = i * dr
          mu = dsqrt(1.d0 - r(i) * r(i))
          ir(i) = 1.d0 - u1*(1.d0-mu) - u2*(1.d0-mu)*(1.d0-mu)

        enddo

      end subroutine

      subroutine run_quad(u1, u2, p, nr, nb)

        implicit none

        ! Benchmark parameters.
        double precision, intent(in) :: u1, u2, p
        integer, intent(in) :: nb, nr

        ! System parameters.
        double precision, dimension(nb) :: b, muo1, mu0, lam
        double precision, dimension(nr) :: r, ir
        double precision :: db
        integer :: i, nrep=10000, us=1e6

        ! Timing.
        integer :: c_s, c_e, rate

        ! Error parameters.
        double precision :: e

        call system_clock(count_rate=rate)

        ! Build an array of impact parameters.
        db = (1.d0 + p) / dble(nb - 1)
        do i=1,nb
          b(i) = dble(i - 1) * db
        enddo

        ! Run the Mandel & Agol routines.
        call system_clock(count=c_s)
        do i=1,nrep
          call occultquad(b, u1, u2, p, muo1, mu0, nb)
        enddo
        call system_clock(count=c_e)

        write(*,*) "Mandel & Agol took:", &
                   real((c_e - c_s)) / rate / nrep * us, &
                   "us"

        ! Build the histogram approximation to the limb-darkening
        ! profile.
        call quadlaw(u1, u2, nr, r, ir)

        ! Run our routine.
        call system_clock(count=c_s)
        do i=1,nrep
          call hist_limb_darkening(p, nr, r, ir, nb, b, lam)
        enddo
        call system_clock(count=c_e)

        write(*,*) "We took:", &
                   real((c_e - c_s)) / rate / nrep * us, &
                   "us"

        ! Compute the error.
        e = 0.d0
        do i=1,nb
          if (muo1(i) .gt. 0.d0) then
            e = e + dsqrt((muo1(i)-lam(i)) * (muo1(i)-lam(i)))/muo1(i)
          endif
        enddo

        write(*,*) "The relative error is:", e

      end subroutine

      program bench

        implicit none

        call run_quad(0.5d0, 0.1d0, 0.1d0, 200, 101)

      end
