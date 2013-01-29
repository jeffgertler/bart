      subroutine lightcurve(n, t, &
                            fstar, mstar, rstar, iobs, &
                            np, r, a, t0, e, pomega, ix, iy, &
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
        ! :param r: (double precision(np))
        !   The sizes of the planets in units of the star's radius.
        !
        ! :param a: (double precision(np))
        !   The semi-major axes of the orbits in units of the star's
        !   radius.
        !
        ! :param t0: (double precision(np))
        !   The time of a reference pericenter passage in days.
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

        implicit none

        double precision :: pi=3.141592653589793238462643d0

        ! The times where the lightcurve should be evaluated.
        integer, intent(in) :: n
        double precision, dimension(n), intent(in) :: t

        ! The properties of the star and the system.
        double precision, intent(in) :: fstar,mstar,rstar,iobs

        ! The planets.
        integer, intent(in) :: np
        double precision, dimension(np), intent(in) :: &
                                          r, a, e, t0, pomega, ix, iy

        ! The limb-darkening profile.
        integer, intent(in) :: nld
        double precision, dimension(nld), intent(in) :: rld, ild

        ! The occulted flux to be calculated.
        double precision, dimension(n), intent(out) :: flux

        ! Status code. Success = 0.
        integer, intent(out) :: info

        integer :: i, j
        double precision, dimension(3, n) :: pos
        double precision, dimension(n) :: b, tmp

        info = 0

        ! Initialize the full lightcurve to the un-occulted stellar
        ! flux.
        flux(:) = fstar

        ! Loop over the planets and solve for their orbits and transit
        ! profiles.
        do i=1, np

          call solve_orbit(n, t, mstar, &
                           e(i), a(i) * rstar, t0(i), pomega(i), &
                           (90.d0 - iobs + ix(i)) / 180.d0 * pi, iy, &
                           pos, info)

          ! Make sure that the orbit was properly solved.
          if (info.ne.0) then
            return
          endif

          b = dsqrt(pos(2,:) * pos(2,:) + pos(3,:) * pos(3,:)) / rstar

          ! HACK: deal with positions behind star.
          do j=1, n
            if (pos(1,j) .le. 0.0d0) then
              b(j) = 1.1d0 + r(i)
            endif
          enddo

          call ldlc(r(i), nld, rld, ild, n, b, tmp)
          flux = flux * tmp

        enddo

      end subroutine
