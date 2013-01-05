      subroutine wt2psi(wt, e, psi)

        ! Solve for the eccentric anomaly given a mean anomaly and an
        ! eccentricity using Halley's method.
        !
        ! :param wt: (double precision)
        !   The mean anomaly.
        !
        ! :param e: (double precision)
        !   The eccentricity of the orbit.
        !
        ! :returns psi: (double precision)
        !   The eccentric anomaly.

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

          ! Take a second order step.
          psi = psi0 - 2.d0 * f * fp / (2.d0 * fp * fp - f * fpp)

          if (abs(psi - psi0) .le. tol) then
            return
          endif

          psi0 = psi

        enddo

        write(*,*) "Warning: root finding didn't converge.", wt, e

      end subroutine

      subroutine solve_orbit(n, t, mstar, e, a, t0, pomega, incl, pos)

        ! Solve Kepler's equations for the 3D position of a point mass
        ! eccetrically orbiting a larger mass.
        !
        ! :param n: (integer)
        !   The number of samples in the time series.
        !
        ! :param t: (double precision(n))
        !   The time series points in days.
        !
        ! :param mstar: (double precision)
        !   The mass of the central body in solar masses.
        !
        ! :param e: (double precision)
        !   The eccentricity of the orbit.
        !
        ! :param a: (double precision)
        !   The semi-major axis of the orbit in solar radii.
        !
        ! :param t0: (double precision)
        !   The time of a reference pericenter passage in days.
        !
        ! :param pomega: (double precision)
        !   The angle between the major axis of the orbit and the
        !   observer in radians.
        !
        ! :param incl: (double precision)
        !   The inclination of the orbit relative to the observer in
        !   radians.
        !
        ! :returns pos: (double precision(3, n))
        !   The output array of positions (x,y,z) in solar radii.
        !   The x-axis points to the observer.

        implicit none

        double precision :: pi=3.141592653589793238462643d0
        double precision :: G=2945.4625385377644d0

        integer, intent(in) :: n
        double precision, dimension(n), intent(in) :: t
        double precision, intent(in) :: mstar,e,a,t0,pomega,incl
        double precision, dimension(3, n), intent(out) :: pos

        integer :: i
        double precision :: period,manom,psi,cpsi,d,cth,r,x,y,xp,yp

        period = 2 * pi * dsqrt(a * a * a / G / mstar)

        do i=1,n

          manom = 2 * pi * (t(i) - t0) / period

          call wt2psi(manom, e, psi)
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
