      subroutine occarea(r0, p, b, area)

        ! Compute the occulted area for a disk of radius `p` in front of
        ! a disk of radius `r0` and impact parameter `b`.

        implicit none

        double precision :: pi=3.141592653589793238462643D0
        double precision, intent(in) :: r0, p, b
        double precision, intent(out) :: area
        double precision :: r2, p2, b2
        double precision :: k1, k2, k3

        if (b .ge. r0 + p) then
          area = 0.0D0
        elseif (b .le. r0 - p) then
          area = pi * p * p
        elseif (b .le. p - r0) then
          area = pi * r0 * r0
        else
          r2 = r0 * r0
          p2 = p * p
          b2 = b * b

          k1 = dacos(0.5D0 * (b2 + p2 - r2) / b / p)
          k2 = dacos(0.5D0 * (b2 + r2 - p2) / b / r0)
          k3 = dsqrt((p+r0-b) * (b+p-r0) * (b-p+r0) * (b+r0+p))

          area = p2 * k1 + r2 * k2 - 0.5 * k3
        endif

      end subroutine

      subroutine ldlc(p, nbins, r, ir, n, b, lam)

        ! Compute the limb-darkened lightcurve for a planet of radius
        ! `p` passing in front of a star (at a set of impact parameters
        ! `b`). The limb-darkening profile of the star is defined by the
        ! samples `ir` at radii `r`.

        implicit none

        double precision :: pi=3.141592653589793238462643D0
        integer :: i, j

        ! The planet radius.
        double precision, intent(in) :: p

        ! The limb-darkening profile specified by `nbins` samples. The
        ! radial bins are expected to be sorted in ascending order.
        integer, intent(in) :: nbins
        double precision, dimension(nbins), intent(in) :: r, ir
        double precision, dimension(nbins) :: r2

        ! The array of impact parameters.
        integer, intent(in) :: n
        double precision, dimension(n), intent(in) :: b

        ! An array of occulted fluxes.
        double precision, dimension(n), intent(out) :: lam

        ! The total flux of the star---used for normalization.
        double precision :: norm

        ! The occulted areas.
        double precision, dimension(nbins) :: areas

        ! First, compute the normalization constant by integrating over
        ! the face of the star.
        r2 = r * r
        norm = r2(1) * ir(1)
        do i=2,nbins
          norm = norm + ir(i) * (r2(i) - r2(i - 1))
        enddo
        norm = pi * norm

        do i=1,n

          ! Then, compute the occulted area in each annulus.
          do j=1,nbins
            call occarea(r(j), p, b(i), areas(j))
          enddo

          ! Do the first order integral over radial bins.
          lam(i) = areas(1) * ir(1)
          do j=2,nbins
            lam(i) = lam(i) + ir(j) * (areas(j) - areas(j - 1))
          enddo
          lam(i) = 1.0D0 - lam(i) / norm

        enddo

      end subroutine
