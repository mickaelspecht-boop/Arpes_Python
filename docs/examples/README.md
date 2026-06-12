# ARPES Logbook Blueprints

These Excel files are small templates for building logbooks that ARPES Explorer
can attach to measurement files.

## Files

- `arpes_logbook_blueprint_simple.xlsx` shows one logbook sheet for one data
  folder. Use this when all files are in the same folder or when a single table
  covers the session.
- `arpes_logbook_blueprint_scoped.xlsx` shows one workbook with several sheets.
  Each sheet declares `Folder Name` near the top, then contains the table for
  that subfolder. Use this when a beamtime folder contains several samples,
  cleaves, or temperature series.

## Required Columns

- `File`: filename, scan number, or distinctive token that appears in the file
  path.
- `hv`: photon energy in eV.

## Useful Optional Columns

- `Temp`: sample temperature in K.
- `Pol`: polarization, for example `LH`, `LV`, `RC`, `LC`, `s`, or `p`.
- `Direction`: cut direction, for example `G-M`, `Gamma-X`, or `G-S`.
- `Azi`, `Polar`, `Tilt`: manipulator angles in degrees.
- `Formula`, `MP-ID`: sample formula and Materials Project identifier.
- `a`, `b`, `c`: lattice constants in angstrom.
- `Work Function`: work function in eV.

Empty `Direction`, `Pol`, and `Azi` cells inherit the previous non-empty value.
This is useful when several consecutive scans share the same geometry.
