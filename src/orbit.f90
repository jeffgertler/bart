      subroutine wt2psi(wt, e, psi)

        ! Solve for the eccentric anomaly given a mean anomaly and an
        ! eccentricity using Halley's method.

        implicit none

        double precision, intent(in) :: wt, e
        double precision, intent(out) :: psi
        double precision :: psi0, f, fp, fpp, tol=1.48e-8
        integer :: it, maxit=100

        psi0 = wt
        do it=1,maxit

          ! Compute the function and derivatives.
          f = psi0 - e * sin(psi0) - wt
          fp = 1.d0 - e * cos(psi0)
          fpp = e * sin(psi0)

          psi = psi0 - 2.d0 * f * fp / (2.d0 * fp * fp - f * fpp)

          if (abs(psi - psi0) .le. tol) then
            return
          endif

          psi0 = psi

        enddo

        write(*,*) "Warning: root finding didn't converge.", wt, e

      end subroutine

      subroutine solve_orbit(n, t, e, a, period, phi, pomega, incl, pos)

        ! Solve Kepler's equations for the 3D position of a point mass
        ! eccetrically orbiting a larger mass.

        implicit none

        double precision :: pi=3.141592653589793238462643D0
        integer, intent(in) :: n
        double precision, dimension(n), intent(in) :: t
        double precision, intent(in) :: e, a, period, phi, pomega, incl
        double precision, dimension(3,n), intent(out) :: pos

        integer :: i
        double precision :: manom, psi, cpsi, d, cth, r, x, y, xp, yp

        do i=1,n

          manom = 2 * pi * t(i) / period - phi

          ! MAGIC: zero tolerance for eccentricity?
          if (e .gt. 1.e-6) then
            call wt2psi(manom, e, psi)
          else
            psi = manom
          endif
          cpsi = dcos(psi)
          d = 1.0d0 - e * cpsi
          cth = (cpsi - e) / d
          r = a * d

          ! In the plane of the orbit.
          x = r * cth
          y = r * dsign(dsqrt(1 - cth * cth), dsin(psi))

          ! Rotate by pomega.
          xp = x * dcos(pomega) + y * dsin(pomega)
          yp = -x * dsin(pomega) + y * dcos(pomega)

          pos(1,i) = xp * dcos(incl)
          pos(2,i) = yp
          pos(3,i) = xp * dsin(incl)

        enddo

      end subroutine
