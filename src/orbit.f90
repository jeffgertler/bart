      subroutine eanom2manom(eanom, e, manom)

        ! Return the mean anomaly given an eccentric anomaly and
        ! eccentricity.

        implicit none

        double precision, intent(in) :: eanom, e
        double precision, intent(out) :: manom

        manom = eanom - e * dsin(eanom)

      end subroutine

      subroutine manom2eanom(manom, e, eanom)

        ! Solve for the eccentric anomaly given a mean anomaly and an
        ! eccentricity.

        implicit none

        double precision, intent(in) :: manom, e
        double precision, intent(out) :: eanom
        double precision :: tmp, delta, tol=1.25e-8
        integer :: it, maxit=100

        eanom = manom + e * dsin(manom)

        do it=1,maxit

          call eanom2manom(eanom, e, tmp)
          delta = manom - tmp
          eanom = eanom + delta / (1.0d0 - e * dcos(eanom))
          if (delta * delta .le. tol) then
            return
          endif

        enddo

        write(*,*) "Warning: root finding didn't converge.", manom, e

      end subroutine

      subroutine solve_orbit(n, t, e, a, period, phi, incl, pos)

        ! Solve Kepler's equations for the 3D position of a point mass
        ! eccetrically orbiting a larger mass.

        implicit none

        double precision :: pi=3.141592653589793238462643D0
        integer, intent(in) :: n
        double precision, dimension(n), intent(in) :: t
        double precision, intent(in) :: e, a, period, phi, incl
        double precision, dimension(3,n), intent(out) :: pos

        integer :: i
        double precision :: manom, psi, cpsi, d, cth, r, x, y

        do i=1,n

          manom = 2 * pi * t(i) / period + phi
          call manom2eanom(manom, e, psi)
          cpsi = dcos(psi)
          d = 1.0d0 - e * cpsi
          cth = (cpsi - e) / d
          r = a * d

          ! In the plane of the orbit.
          x = r * cth
          y = r * dsign(dsqrt(1 - cth * cth), dsin(psi))

          pos(1,i) = x * dcos(incl)
          pos(2,i) = y
          pos(3,i) = x * dsin(incl)

        enddo

      end subroutine
