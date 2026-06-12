# Physics Notes

These notes summarize the assumptions behind the reported numbers.

## Energy Axis

Band maps are displayed as E - EF in eV. The EF offset shifts the measured energy axis so the Fermi level sits at zero. An Au reference can usually be reused within the same stable acquisition context. Manual offsets should be noted.

## Momentum Axis

Momentum axes depend on photon energy, geometry, work function, lattice constants, and angle convention. pi/a is convenient for Brillouin-zone work. A^-1 is better when comparing samples with different lattice constants.

Browse-only maps skip this conversion when metadata is incomplete.

## Gamma

Gamma is the symmetry center used by overlays, cuts, and paired MDC fits. A wrong Gamma shifts more than the picture: it changes symmetric peak pairs, pocket areas, and DFT comparisons. Recheck it after distortion correction or when switching between FS-derived and BM-derived references.

## MDC Fits

An MDC is intensity versus k at fixed energy. The fitter models each MDC with Lorentzian peak pairs around a common center. The resulting kF points form the experimental dispersion.

Window choice matters. Too wide, and unrelated bands can pull the fit. Too narrow, and shoulders, asymmetry, or background problems can disappear from view.

## Width and Resolution

The fitted linewidth contains both instrumental and physical contributions. Corrected Gamma uses available energy and momentum resolution estimates. Be careful when the resolution source is missing, defaulted, or guessed manually.

## Fermi Velocity and Effective Mass

vF comes from the local slope of the fitted dispersion near EF. Effective mass uses kF and vF after unit conversion. Very large or unstable masses usually point to a nonlinear near-EF dispersion, a poor fit window, or uncertain kF.

## FS Pockets

Pocket area depends on contour level, BZ definition, Gamma center, and distortion correction. Auto level is a starting point; the contour still needs a visual check. For noisy maps, stable presets may be more reliable than fine presets.

## DFT Overlays and Re Sigma

DFT bands are references. They still need the right material, path, band index, energy shift, and momentum alignment. Re Sigma compares the experimental dispersion with an interpolated DFT band, so those choices matter.

## KZ

kz conversion uses photon energy, work function, inner potential V0, lattice c, and the free-electron final-state model. The KZ tab can show converted k// vs kz or raw k// vs hv. Check the raw view before interpreting the converted one.

Use fitted V0 as an estimate, especially when the hv range shows clear periodic structure.

## Reliability Checks

Check raw intensity before processed contrast, calibrated axes before derived quantities, and repeated behavior before single-slice fits.
