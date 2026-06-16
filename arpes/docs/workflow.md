# Workflow

The usual path is simple: load the data, set the sample context, check the axes, then fit. Most mistakes later in the analysis come from skipping one of those early checks.

## 1. Start With the Session

Open the measurement folder from the browser. The session is saved next to that folder, so loaded files, fit parameters, notes, tags, EF offsets, sample settings, and results are restored when you come back.

Attach a global logbook when one sheet describes the whole folder. Attach scoped logbooks when different subfolders are different samples or cleaves. Metadata is taken from the file first, then from the logbook, then from manual fields when the file format is incomplete.

For large folders, group the browser by folder, type, photon energy, temperature, polarization, lab, or path. After a long session, "Loaded only" cuts the list down to files that already have analysis state.

## 2. Describe the Sample Before Analysis

When Sample setup opens, fill the values that define the coordinate system: work function, lattice constants, and cut direction when it is known. These fields drive k axes, FS cuts, kz maps, BZ overlays, and effective masses.

If the metadata is incomplete and you only need to inspect the file, choose Browse only. The map stays on raw theta/E axes, so the app does not invent a momentum calibration.

## 3. Inspect the Band Map

Start in the BM tab. Raw is the reference view; EDC normalization, second derivative, and curvature are contrast views. Click the map to inspect the current MDC and EDC. Display gamma only changes visibility, not the stored data.

Check EF before fitting. With an Au reference, run automatic EF calibration, save the reference, then apply it to files from the same acquisition context. You can still view uncalibrated files, but near-EF fits should wait.

## 4. Make the Momentum Origin Explicit

Set Gamma from the clearest view. On an FS map, use the FS Gamma tools or pick the center manually. On a BM cut, use Auto Gamma BM when the cut itself shows the symmetry point. The status badge shows whether a Gamma reference is active and which axis state it belongs to.

For FS data, set the BZ shape, lattice, orientation, and half-BZ controls before measuring pockets. Apply distortion correction only when the map geometry calls for it, then recheck the FS and BM overlays.

The FFT grid filter is only for visible detector artifacts. It does not rewrite the raw data and should not hide a bad fit window.

## 5. Explore the FS Volume

With a calibrated FS volume, FS Explorer lets you browse cuts before fitting them. The left view is the iso-energy FS map; the line is the cut extracted as a BM on the right. Drag the center to move it, drag an end handle to rotate or resize it, or set the angle with the spin controls. Play sweeps through directions.

FS Explorer is disabled on raw Browse-only axes. Free cuts need calibrated k axes; otherwise the angle display would be misleading.

## 6. Fit MDCs Deliberately

Move to MDC Fit once the BM cut, EF, and Gamma are in good shape. Define the energy and k window first. Pick the number of Lorentzian peak pairs, run an estimate at the current energy, and check whether the initial peaks sit on the visible bands.

Fit zones are for files that need several independent windows, branches, or parameter sets. In the Zones MDC table, add a zone from the current range, then select its row to load and edit its parameters; the selected zone follows your analysis range as you adjust it, so refining a window updates that zone in place. Run all fits every active zone in one pass. Each zone keeps its own parameters and result, and the active zone is the one shown by legacy plots and exports.

Run the full fit after the estimate looks sane. Then check Waterfall and EDC views to see whether the model follows the data slice by slice. Bad points can be selected, deleted, undone, redone, and annotated.

Batch fit works best after one representative file has been tuned by hand. Propagate parameters only across files with the same geometry, energy range, and band topology.

## 7. Check Results Before Reporting

The Results tab gathers the fitted files. The dispersion plot shows kF(E); the lifetime view shows Gamma(E). In the per-slice table, large center drift, broad corrected Gamma, or high reduced chi2 usually points back to the ROI, Gamma centering, pair count, or EF calibration.

The physical table contains the values you would report: kF, vF, effective mass, and Gamma0 with uncertainties. Bootstrap sigma is useful when a few remaining outliers would make propagated errors too optimistic.

## 8. Add Higher-Level Analysis

Open Band Analysis after the MDC results are stable. TB fit gives a compact dispersion model, Kink Sigma compares the fitted dispersion with a reference band, and Gap analysis follows symmetrized or leading-edge behavior when the data supports it.

Treat DFT overlays as references. Local imports and Materials Project bands still need manual alignment, band choice, and a check against the measured direction.

## 9. Build KZ Maps From Variable-hv Series

The KZ tab is for folders where photon energy changes across scans. Load a KZ logbook if hv is not reliable in the files. Tune or fit V0 after lattice c and work function are set. The raw k// vs hv view is useful before interpreting the free-electron kz conversion.

## 10. Finish With Notes, Export, and Reproducibility

Write down the choices that will matter later: calibration source, rejected points, chosen zones, DFT alignment, and export settings. Save the session before sharing results.

Results export writes CSV, TXT, LaTeX tables, provenance sidecars, and publication-style figures. FS export is for maps where the overlays are part of the figure. Export after checking the fits, calibration, and overlays.
