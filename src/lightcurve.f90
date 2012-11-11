      subroutine rj2rs(rj, rs)

        ! Convert Jupiter radii to Solar radii.

        implicit none

        double precision, intent(in) :: rj
        double precision, intent(out) :: rs
        rs = 9.94493d-2 * rj

      end subroutine

      subroutine au2rs(au, rs)

        ! Convert AU to Solar radii.

        implicit none

        double precision, intent(in) :: au
        double precision, intent(out) :: rs
        rs = 2.150856d2 * au

      end subroutine

      subroutine lightcurve(n, t, &
                            rs, fs, iobs, &
                            np, rp, ap, ep, tp, php, ip, &
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
        ! :param rs: (double precision)
        !   The radius of the star in Solar radii.
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
        !   The sizes of the planets in Jupiter radii.
        !
        ! :param ap: (double precision(np))
        !   The semi-major axes of the orbits in AU.
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
        double precision, intent(in) :: rs, fs, iobs

        ! The planets.
        integer, intent(in) :: np
        double precision, dimension(np), intent(in) :: &
                                              rp,ap,ep,tp,php,ip

        ! The limb-darkening profile.
        integer, intent(in) :: nld
        double precision, dimension(nld), intent(in) :: rld, ild

        ! The occulted flux to be calculated.
        double precision, dimension(n), intent(out) :: flux

        integer :: i, j
        double precision, dimension(3,n) :: pos
        double precision, dimension(n) :: b, tmp
        double precision :: a, r

        ! Initialize the full lightcurve to the un-occulted stellar
        ! flux.
        flux(:) = fs

        ! Loop over the planets and solve for their orbits and transit
        ! profiles.
        do i=1,np

          call au2rs(ap(i), a)
          call solve_orbit(n, t, &
                           ep(i), a, tp(i), php(i), &
                           (iobs+ip(i)) / 180.d0 * pi, pos)

          b = dsqrt(pos(2,:) * pos(2,:) + pos(3,:) * pos(3,:)) / rs

          ! HACK: deal with positions behind star.
          do j=1,n
            if (pos(1,j) .le. 0.0d0) then
              b(j) = 1.1d0 + rp(i)
            endif
          enddo

          call rj2rs(rp(i), r)
          call ldlc(r / rs, nld, rld, ild, n, b, tmp)
          flux = flux * tmp

        enddo

      end subroutine
