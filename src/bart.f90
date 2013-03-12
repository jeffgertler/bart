      program bart

        implicit none

        integer :: i, parsed, nplanets, nldp, nt, usequad, info
        character(len=100) :: arg

        double precision :: g1, g2, texp, tread
        double precision, allocatable :: bins(:), intensity(:), m(:), &
                                         r(:), a(:), t0(:), e(:), &
                                         po(:), ix(:), iy(:), t(:), &
                                         flux(:), tmpt(:)

        double precision :: a2, b2, k1, k2, k3, th

        ! Defaults
        nplanets = 1
        nldp = 50
        texp = 1626./60./60./24.

        ! Loop over command line arguments.
        usequad = 0
        parsed = 1
        i = 1
        do

          call get_command_argument(i, arg)
          if (len_trim(arg) == 0) exit

          if (arg .eq. "-n") then

            ! Parse the number of planets.
            i = i + 1
            call get_command_argument(i, arg)
            if (len_trim(arg) == 0) then
              parsed = 0
              exit
            endif
            read (arg,*) nplanets

          elseif (arg .eq. "-k") then

            ! Parse the number of limb darkening bins.
            i = i + 1
            call get_command_argument(i, arg)
            if (len_trim(arg) == 0) then
              parsed = 0
              exit
            endif
            read (arg,*) nldp

          elseif (arg .eq. "--texp") then

            ! Parse the exposure time.
            i = i + 1
            call get_command_argument(i, arg)
            if (len_trim(arg) == 0) then
              parsed = 0
              exit
            endif
            read (arg,*) texp

          elseif (arg .eq. "-g") then

            ! Parse the gammas.
            usequad = 1

            ! Gamma 1.
            i = i + 1
            call get_command_argument(i, arg)
            if (len_trim(arg) == 0) then
              parsed = 0
              exit
            endif
            read (arg,*) g1

            ! Gamma 2.
            i = i + 1
            call get_command_argument(i, arg)
            if (len_trim(arg) == 0) then
              parsed = 0
              exit
            endif
            read (arg,*) g2

          elseif (arg .eq. "-h" .or. arg .eq. "--help") then

            parsed = 0
            exit

          else

            write (*,*) "Unrecognized argument '", trim(arg), "'"
            parsed = 0
            exit

          endif

          i = i + 1

        enddo

        if (parsed .eq. 0) then
          write (*,*) "Usage:"
          write (*,*) "  bart [-h | --help]"
          write (*,*) "  bart [-n NPLANETS] [-k NLDP] [-t NTIME] [-g ",&
                      "G1 G2] [--texp TEXP]"
          write (*,*)
          write (*,*) "-h, --help    show this message"
          write (*,*) "-n NPLANETS   the number of planets in the ", &
                      "system [default: 1]"
          write (*,*) "-k NLDP       the number of bins in the limb ", &
                      "darkening profile [default: 50]"
          write (*,*) "-g G1 G2      coefficients to use a quadratic ",&
                      "limb darkening law"
          write (*,*) "--texp TEXP   the exposure time in days ",&
                      "[default 1626./60./60./24.]"
          write (*,*)
          write (*,*) "Then write one line for each planet to stdin ", &
                      "with the form:"
          write (*,*) "mass, radius, semi-major axis, transit time, ",&
                      "eccentricity, pomega, incl_x, incl_y"
          stop
        endif

        ! Allocate the arrays.
        allocate(m(nplanets))
        allocate(r(nplanets))
        allocate(a(nplanets))
        allocate(t0(nplanets))
        allocate(e(nplanets))
        allocate(po(nplanets))
        allocate(ix(nplanets))
        allocate(iy(nplanets))

        ! Read in the planet parameters.
        do i=1,nplanets

          read (*,*) m(i), r(i), a(i), t0(i), e(i), po(i), ix(i), iy(i)

        enddo


        allocate(bins(nldp))
        allocate(intensity(nldp))

        ! Build or read the limb darkening profile.
        if (usequad .eq. 1) then

          do i=1,nldp

            ! Compute the integrated intensity in the bins.
            a2 = dble(i - 1) * (i - 1) / nldp / nldp
            b2 = dble(i * i) / nldp / nldp
            th = 0.5 * 3
            k1 = 0.5 * (b2 - a2) * (1 - g1 - 2 * g2)
            k2 = (g1 + 2 * g2) * ((1 - a2) ** th - (1 - b2) ** th) / 3
            k3 = 0.25 * g2 * (b2 * b2 - a2 * a2)
            bins(i) = dble(i) / nldp
            intensity(i) = 2 * (k1 + k2 + k3) / (b2 - a2)

          enddo

        else

          write (*,*) "Not implemented"
          stop

        endif

        ! Read the time samples until EOF.
        nt = 0
        do

          read (*,*,iostat=info) tread

          if (info > 0) then
            write (*,*) "Couldn't read input"
            stop
          elseif (info < 0) then
            exit
          else
            nt = nt + 1

            ! Resize the time array.
            if (nt .eq. 1) then
              allocate(t(nt))
              t(nt) = tread
            else
              allocate(tmpt(nt))
              tmpt(1:size(t)) = t
              tmpt(nt) = tread
              deallocate(t)
              allocate(t(size(tmpt)))
              t = tmpt
              deallocate(tmpt)
            endif

          endif

        enddo

        ! Compute the light curve.
        info = 0
        allocate(flux(nt))
        call lightcurve(nt, t, texp, 3, 1.d0, 1.d0, 1.d0, 90.d0, &
                        nplanets, m, r, a, t0, e, po, ix, iy, &
                        nldp, bins, intensity, flux, info)

        ! Print the results to stdout.
        do i=1,nt
          write (*,*) t(i), flux(i)
        enddo

      end program
