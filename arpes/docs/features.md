# Features

Use this page as a quick map of the main ARPES Explorer tools.

## Browser and Sessions

The left browser is where the experiment folder lives. Double-click a file to load it, move through neighboring files with the navigation shortcuts, and tag useful groups such as "good", "Au", "FS", or "redo".

Sessions keep the working state beside the data folder: loaded entries, sample metadata, EF offsets, fit parameters, fit zones, notes, annotations, and results. Recent sessions and session comparison are in the File menu.

For beamtime folders, group by folder, file type, photon energy, temperature, path, polarization, or lab. The loaded-only filter shows files that already have analysis state.

## Logbooks and Sample Setup

Logbooks fill in metadata that raw beamline files often miss: photon energy, temperature, direction, sample name, and lattice values. A global logbook applies to the folder. A scoped logbook applies only to a subfolder, useful when one acquisition folder contains several samples.

Sample setup defines the physical coordinate system: work function, lattice constants, and cut direction. Those values feed k conversion, FS overlays, kz conversion, and mass estimates.

## Loaders and Browse Only

The loaders read several ARPES formats through the same pipeline. When the file and logbook provide enough metadata, the app builds calibrated energy and momentum axes. When they do not, Browse only keeps the raw axes so the file can still be inspected.

The cache stores loaded raw artifacts on disk for fast reloads. "Reload current file (no cache)" forces a fresh read after source files or logbooks change.

## BM Tab

BM is the main band-map view. Raw intensity shows the data as loaded. EDC normalization helps compare dispersions across energy. Second derivative and curvature sharpen bands, but they are contrast views; confirm important features in Raw or EDCnorm.

Click the map to update the current MDC and EDC before choosing the fit window and number of peak pairs.

## EF Calibration

EF calibration moves the displayed energy zero to the measured Fermi level. With an Au reference, save the reference and apply it to matching files. Manual EF offsets are allowed; note the source when it affects final numbers.

## Distortion and Grid Corrections

Distortion controls preview and apply trapezoid or parabolic detector corrections. Check the live preview before applying, and propagate only when the same correction fits the target files.

The grid filter removes visible FFT-like detector artifacts from the displayed BM. It helps with viewing; it does not make a poor raw fit good.

## Gamma and Momentum Centering

Gamma tools set the symmetry center in the current coordinate system. FS-derived Gamma works when the FS map contains the center. BM-derived Gamma is better when the cut itself shows the symmetry point. Manual picks are fine for difficult data; record them in Notes.

Overlays and fits remember whether they were made with distorted or grid-corrected axes. If the axis state changes, stale overlays are rejected.

## FS Tab

The FS tab shows Fermi-surface maps, BZ overlays, Gamma markers, distortion controls, and pocket measurements. Set the geometry here before extracting many BM cuts. Pocket areas depend on the contour level, BZ setup, and distortion correction.

Compatible BM cuts can be overlaid on the FS map. Exact matches, rotated candidates, and scaled candidates use different styles, so mismatched geometry is easier to spot.

## FS Explorer

FS Explorer is an interactive cut browser for FS volumes. The map view shows an iso-energy slice, the line defines the cut, and the right view shows the extracted BM. Drag the line, set the angle, change energy, or sweep through directions.

Use it to choose cut directions before running full MDC fits. It requires calibrated k axes, so raw Browse-only maps are excluded.

## MDC Fit

MDC Fit extracts dispersions by fitting k profiles at fixed energy. The model uses symmetric Lorentzian peak pairs around a center. Start with a single-energy estimate, check the peak placement, then run the full energy range.

Fit zones let one file carry several independent analyses: different pockets, k windows, pair counts, or conservative versus broader fits.

Batch fitting applies the tuned MDC setup to multiple files. Use it only after one representative file fits cleanly.

## Waterfall and EDC Views

Waterfall stacks MDCs and overlays fitted kF positions. It quickly shows whether the model follows the measured peaks through energy. EDC shows energy profiles at fixed k, catching features an MDC-only pass can miss.

## Results

Results shows fit diagnostics, physical tables, multi-file plots, and export controls. The first table is fit quality per file and slice. The physical table is where kF, vF, m*, and Gamma0 are reported with uncertainties.

File filters narrow comparisons to related cuts. "Recompute physical results" updates the tables after point deletion. Bootstrap sigma helps when the remaining point cloud is uneven or has mild outliers.

## Band Analysis

Band Analysis starts from completed MDC fits. The Summary view shows what is ready and what still needs setup. TB fit models the dispersion. Kink Sigma compares the measured dispersion with a reference band. Gap analysis is for gap-like features when the acquisition and symmetry support it.

## DFT and Materials Project Overlays

DFT overlays are references, not calibration. Import local bands or query Materials Project, then choose the band and alignment manually. The overlay can guide orientation, Gamma expectations, and self-energy comparisons.

## KZ

KZ maps combine a variable-photon-energy BM series into k// vs kz or raw k// vs hv views. The converted kz view uses work function, inner potential, lattice constants, and the free-electron final-state approximation. The raw hv view is for checking the scan before interpreting the conversion.

## Notes and Annotations

Notes store calibration choices, rejected points, beamline issues, and export decisions. Annotations attach local comments to fit points or map features.

## Export

Export writes result tables, physical summaries, provenance sidecars, and figures. Figure presets provide common plot layouts. Provenance files keep the inputs behind the numbers.

FS maps can also be exported with overlays when the BZ, pockets, or compatible BM cuts are part of the result.
