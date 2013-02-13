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
        !
        ! :returns info: (integer)
        !   A flag that indicates whether or not a solution was
        !   successfully found. (``0 == success``)

        implicit none

        double precision, intent(in) :: wt, e
        double precision, intent(out) :: psi
        integer, intent(out) :: info
        double precision :: psi0, spsi0, f, fp, fpp, tol=1.48e-8
        integer :: it, maxit=100

        info = 0

        psi0 = wt
        do it=1,maxit

          ! Compute the function and derivatives.
          spsi0 = sin(psi0)
          f = psi0 - e * spsi0 - wt
          fp = 1.d0 - e * cos(psi0)
          fpp = e * spsi0

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

      subroutine velocity_amplitude(mstar, mplanet, incl, e, P, K)

        ! Compute the velocity amplitude of an orbit given the orbital
        ! elements and the mass of the star and planet.
        !
        ! :param mstar: (double precision)
        !   The mass of the star in Solar masses.
        !
        ! :param mplanet: (double precision)
        !   The mass of the planet in Solar masses.
        !
        ! :param incl: (double precision)
        !   The inclination of the orbit relative to the line-of-sight
        !   in radians. ``0.5 * pi`` indicates an edge-on viewing angle.
        !
        ! :param e: (double precision)
        !   The eccentricity of the orbit.
        !
        ! :param P: (double precision)
        !   The period of the orbit in days.
        !
        ! :returns K: (double precision)
        !   The velocity amplitude in Solar radii per day.

        double precision, intent(in) :: mstar, mplanet, incl, e, P
        double precision, intent(out) :: K
        double precision :: pi=3.141592653589793238462643d0
        ! Newton's constant in R_sun^3 M_sun^-1 days^-2.
        double precision :: G=2945.4625385377644d0

        K = (2*pi*G/P)**(1.d0/3)
        K = K * mplanet * dsin(incl) * (mplanet + mstar) ** (-2.d0/3)
        K = K / dsqrt(1 - e * e)

      end subroutine

      subroutine solve_orbit(n, t, mstar, mplanet, e, a, t0, pomega, &
                             ix, iy, rvflag, pos, radvel, info)

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
        ! :param mplanet: (double precision)
        !   The mass of the planet in solar masses.
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
        ! :param rvflag: (integer)
        !   Should we compute the radial velocity? 0: no, 1: yes.
        !
        ! :returns pos: (double precision(3, n))
        !   The output array of positions (x,y,z) in solar radii.
        !   The x-axis points to the observer.
        !
        ! :returns radvel: (double precision(n))
        !   The radial velocity profile in Solar radii per day.
        !
        ! :returns info: (integer)
        !   Was the computation successful? If ``info==0``, it was.
        !   Otherwise, no solution was found for Kepler's equation.

        implicit none

        ! Newton's constant in R_sun^3 M_sun^-1 days^-2.
        double precision :: G=2945.4625385377644d0
        double precision :: pi=3.141592653589793238462643d0

        integer, intent(in) :: n, rvflag
        double precision, dimension(n), intent(in) :: t
        double precision, intent(in) :: mstar,mplanet,e,a,t0,pomega,ix,&
                                        iy
        double precision, dimension(3, n), intent(out) :: pos
        double precision, dimension(n), intent(out) :: radvel

        integer, intent(out) :: info

        integer :: i
        double precision :: period,manom,psi,cpsi,d,cth,r,x,y,xp,yp,&
                            xsx,psi0,t1,K,th,spsi

        period = 2 * pi * dsqrt(a * a * a / G / (mstar + mplanet))

        if (rvflag.eq.1) then
          call velocity_amplitude(mstar,mplanet,0.5*pi-ix,e,period,K)
        endif

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
          spsi = dsin(psi)
          d = 1.0d0 - e * cpsi
          cth = (cpsi - e) / d

          if (rvflag.eq.0) then
            ! In the plane of the orbit.
            r = a * d
            x = r * cth
            y = r * dsign(dsqrt(1 - cth * cth), spsi)

            ! Rotate by pomega.
            xp = x * dcos(pomega) + y * dsin(pomega)
            yp = -x * dsin(pomega) + y * dcos(pomega)

            ! Rotate by the inclination angles.
            xsx = xp * dsin(ix)
            pos(1,i) = xp * dcos(ix)
            pos(2,i) = yp * dcos(iy) - xsx * dsin(iy)
            pos(3,i) = yp * dsin(iy) + xsx * dcos(iy)
          endif

          ! Compute the radial velocity.
          if (rvflag.eq.1) then
            th = dacos(cth)
            th = th * dsign(1.d0, dsin(th)) * dsign(1.d0, spsi)
            radvel(i) = K*(dsin(th-pomega) - e*dsin(pomega))
          endif

        enddo

      end subroutine
