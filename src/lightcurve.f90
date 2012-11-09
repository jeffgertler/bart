      subroutine lightcurve(n, t, &
                            rs, ms, iobs, &
                            np, rp, ap, ep, tp, php, ip, &
                            nld, rld, ild, &
                            lam)

        ! The times where the lightcurve should be evaluated.
        integer, intent(in) :: n
        double precision, dimension(n), intent(in) :: t

        ! The properties of the star and the system.
        double precision, intent(in) :: rs, ms, iobs

        ! The planets.
        integer, intent(in) :: np
        double precision, dimension(np), intent(in) :: rp,ap,ep,tp,php,ip

        ! The limb-darkening profile.
        integer, intent(in) :: nld
        double precision, dimension(nld), intent(in) :: rld, ild

        ! The occulted flux to be calculated.
        double precision, dimension(n), intent(out) :: lam

        integer :: i, j
        double precision, dimension(3,n) :: pos
        double precision, dimension(n) :: b, tmp

        lam(:) = ms

        do i=1,np

          call coords(n, t, ep(i), ap(i)/rs, tp(i), php(i), iobs+ip(i),&
                      pos)

          b = dsqrt(pos(2,:) * pos(2,:) + pos(3,:) * pos(3,:))

          ! HACK: deal with positions behind star.
          do j=1,n
            if (pos(1,j) .le. 0.0d0) then
              b(j) = 1.1d0 + rp(i)
            endif
          enddo

          call hist_limb_darkening(rp(i) / rs, nld, rld, ild, n, b, tmp)
          lam = lam * tmp

        enddo

      end subroutine
