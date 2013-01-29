      subroutine wt2psi(wt, e, psi, info)

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
        integer, intent(out) :: info
        double precision :: psi0, f, fp, fpp, tol=1.48e-8
        integer :: it, maxit=100

        info = 0

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
        info = 1

      end subroutine

      subroutine solve_orbit(n, t, mstar, e, a, t0, pomega, ix, iy, pos, info)

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
        ! :param ix: (double precision)
        !   The inclination of the orbit relative to the observer in
        !   radians.
        !
        ! :param iy: (double precision)
        !   The inclination of the orbit in the plane of the sky in
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
        double precision, intent(in) :: mstar,e,a,t0,pomega,ix,iy
        double precision, dimension(3, n), intent(out) :: pos

        integer, intent(out) :: info

        integer :: i
        double precision :: period,manom,psi,cpsi,d,cth,r,x,y,xp,yp,xsx,&
                            psi0,t1

        period = 2 * pi * dsqrt(a * a * a / G / mstar)
        psi0 = 2 * datan2(dtan(0.5 * pomega), dsqrt((1 + e) / (1 - e)))
        t1 = t0 -  0.5 * period * (psi0 - e * dsin(psi0)) / pi
        info = 0

        do i=1,n

          manom = 2 * pi * (t(i) - t1) / period

          call wt2psi(manom, e, psi, info)
          if (info.ne.0) then
            return
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

          ! Rotate by the inclination angles.
          xsx = xp * dsin(ix)
          pos(1,i) = xp * dcos(ix)
          pos(2,i) = yp * dcos(iy) - xsx * dsin(iy)
          pos(3,i) = yp * dsin(iy) + xsx * dcos(iy)

        enddo

      end subroutine
