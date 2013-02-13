      subroutine lightcurve(n, t, texp, nbin, &
                            fstar, mstar, rstar, iobs, &
                            np, mass, r, a, t0, e, pomega, ix, iy, &
                            nld, rld, ild, &
                            flux, info)

        ! Compute the lightcurve for a planetary system.
        !
        ! :param n: (integer)
        !   The number of points in the lightcurve.
        !
        ! :param t: (double precision(n))
        !   The times where the lightcurve should be evaluated in days.
        !
        ! :param texp: (double precision)
        !   The exposure time in days.
        !
        ! :param nbin: (integer)
        !   The number of bins to integrate for an exposure time.
        !
        ! :param fstar: (double precision)
        !   The un-occulted flux of the star in arbitrary units.
        !
        ! :param mstar: (double precision)
        !   The mass of the star in Solar masses.
        !
        ! :param rstar: (double precision)
        !   The radius of the star in Solar radii.
        !
        ! :param iobs: (double precision)
        !   The observation angle in degrees (90deg corresponds to edge
        !   on).
        !
        ! :param np: (integer)
        !   The number of planets in the system.
        !
        ! :param mass: (double precision(np))
        !   The masses of the planets in Solar masses.
        !
        ! :param r: (double precision(np))
        !   The sizes of the planets in Solar radii.
        !
        ! :param a: (double precision(np))
        !   The semi-major axes of the orbits in Solar radii.
        !
        ! :param t0: (double precision(np))
        !   The time of a reference transit in days.
        !
        ! :param e: (double precision(np))
        !   The eccentricities of the orbits.
        !
        ! :param pomegas: (double precision(np))
        !   The pomegas of the orbits in radians.
        !
        ! :param ix: (double precision(np))
        !   The relative inclinations of the orbits around the axis
        !   perpendicular to the line-of-sight in degrees.
        !
        ! :param iy: (double precision(np))
        !   The relative inclinations of the orbits around the
        !   line-of-sight axis in degrees.
        !
        ! :param nld: (integer)
        !   The number of radial bins in the limb-darkening profile.
        !
        ! :param rld: (double precision(nld))
        !   The positions (in units of the stars radius) of the radial
        !   bins in the limb-darkening profile. WARNING: the radii are
        !   expected to be sorted and things will probably blow up if
        !   they're not. Also, the first bin should be at ``r_1 > 0``
        !   and the final bin should be at ``r_n = 1.0``.
        !
        ! :param ild: (double precision(nld))
        !   The limb-darkening function evaluated at each ``rld``. The
        !   units are arbitrary.
        !
        ! :returns flux: (double precision(n))
        !   The observed flux at each time ``t`` in the same units as
        !   the input ``fstar``.
        !
        ! :returns info: (integer)
        !   Was the computation successful? If ``info==0``, it was.
        !   Otherwise, no solution was found for Kepler's equation.

        implicit none

        double precision :: pi=3.141592653589793238462643d0

        ! The times where the lightcurve should be evaluated.
        integer, intent(in) :: n
        double precision, dimension(n), intent(in) :: t

        ! Exposure time integration.
        double precision, intent(in) :: texp
        integer, intent(in) :: nbin

        ! The properties of the star and the system.
        double precision, intent(in) :: fstar,mstar,rstar,iobs

        ! The planets.
        integer, intent(in) :: np
        double precision, dimension(np), intent(in) :: &
                                      mass, r, a, e, t0, pomega, ix, iy

        ! The limb-darkening profile.
        integer, intent(in) :: nld
        double precision, dimension(nld), intent(in) :: rld, ild

        ! The occulted flux to be calculated.
        double precision, dimension(n), intent(out) :: flux

        ! Status code. Success = 0.
        integer, intent(out) :: info

        integer :: i, j
        double precision, dimension(3, n*nbin) :: pos
        double precision, dimension(n*nbin) :: b, tmp, ttmp, ftmp, rvtmp

        info = 0

        ! Add extra time slices for integration over exposure time.
        do i=1,n
          do j=1,nbin

            ttmp((i-1)*nbin+j) = t(i) + texp * ((j-0.5)/float(nbin) - 0.5)

          enddo
        enddo

        ! Initialize the full light curve to the un-occulted stellar
        ! flux.
        ftmp(:) = fstar

        ! Loop over the planets and solve for their orbits and transit
        ! profiles.
        do i=1, np

          call solve_orbit(n*nbin, ttmp, mstar, mass(i), &
                           e(i), a(i), t0(i), pomega(i), &
                           (90.d0 - iobs + ix(i)) / 180.d0 * pi, &
                           iy / 180.d0 * pi, &
                           pos, rvtmp, info)

          ! Make sure that the orbit was properly solved.
          if (info.ne.0) then
            return
          endif

          b = dsqrt(pos(2,:) * pos(2,:) + pos(3,:) * pos(3,:)) / rstar

          ! HACK: deal with positions behind star.
          do j=1,n*nbin
            if (pos(1,j) .le. 0.0d0) then
              b(j) = 1.1d0 + r(i)
            endif
          enddo

          call ldlc(r(i) / rstar, nld, rld, ild, n*nbin, b, tmp)

          ftmp = ftmp * tmp

        enddo

        ! "Integrate" over exposure time.
        flux(:) = 0
        do i=1,n
          do j=1,nbin
            flux(i) = flux(i) + ftmp((i-1)*nbin + j) / float(nbin)
          enddo
        enddo

      end subroutine
