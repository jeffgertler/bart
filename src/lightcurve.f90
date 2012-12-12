      subroutine lightcurve(n, t, &
                            fs, iobs, &
                            np, rp, ap, ep, tp, php, pop, ip, &
                            nld, rld, ild, &
                            flux)

        ! Compute the lightcurve for a planetary system.
        !
        ! :param n: (integer)
        !   The number of points in the lightcurve.
        !
        ! :param t: (double precision(n))
        !   The times where the lightcurve should be evaluated.
        !
        ! :param fs: (double precision)
        !   The un-occulted flux of the star.
        !
        ! :param iobs: (double precision)
        !   The observation angle in degrees.
        !
        ! :param np: (integer)
        !   The number of planets in the system.
        !
        ! :param rp: (double precision(np))
        !   The sizes of the planets in units of the star's radius.
        !
        ! :param ap: (double precision(np))
        !   The semi-major axes of the orbits in units of the star's
        !   radius.
        !
        ! :param ep: (double precision(np))
        !   The eccentricities of the orbits.
        !
        ! :param tp: (double precision(np))
        !   The periods of the orbits in days.
        !
        ! :param php: (double precision(np))
        !   The phases of the orbits in radians.
        !
        ! :param pop: (double precision(np))
        !   The pomegas of the orbits in radians.
        !
        ! :param ip: (double precision(np))
        !   The inclinations of the orbits in degrees.
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
        !   the input ``fs``.

        implicit none

        double precision :: pi=3.141592653589793238462643d0

        ! The times where the lightcurve should be evaluated.
        integer, intent(in) :: n
        double precision, dimension(n), intent(in) :: t

        ! The properties of the star and the system.
        double precision, intent(in) :: fs, iobs

        ! The planets.
        integer, intent(in) :: np
        double precision, dimension(np), intent(in) :: &
                                              rp,ap,ep,tp,php,pop,ip

        ! The limb-darkening profile.
        integer, intent(in) :: nld
        double precision, dimension(nld), intent(in) :: rld, ild

        ! The occulted flux to be calculated.
        double precision, dimension(n), intent(out) :: flux

        integer :: i, j
        double precision, dimension(3,n) :: pos
        double precision, dimension(n) :: b, tmp

        ! Initialize the full lightcurve to the un-occulted stellar
        ! flux.
        flux(:) = fs

        ! Loop over the planets and solve for their orbits and transit
        ! profiles.
        do i=1,np

          call solve_orbit(n, t, &
                           ep(i), ap(i), tp(i), php(i), pop(i), &
                           (90.d0 - iobs - ip(i)) / 180.d0 * pi, pos)

          b = dsqrt(pos(2,:) * pos(2,:) + pos(3,:) * pos(3,:))

          ! HACK: deal with positions behind star.
          do j=1,n
            if (pos(1,j) .le. 0.0d0) then
              b(j) = 1.1d0 + rp(i)
            endif
          enddo

          call ldlc(rp(i), nld, rld, ild, n, b, tmp)
          flux = flux * tmp

        enddo

      end subroutine
