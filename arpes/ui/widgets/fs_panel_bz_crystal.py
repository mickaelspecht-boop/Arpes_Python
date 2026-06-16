"""BZ crystal overlay drawer for FermiSurfaceCanvas (free functions)."""
from __future__ import annotations


def overlay_bz_crystal(canvas, p, raw_data) -> None:
    """Overlay the crystal BZ polygon and HS labels from the MP lattice.

    Reads ``raw_data["metadata"]["fs_lattice"]`` (MP cache dict) if present.
    Without an MP lattice, draws nothing: a heuristic polygon would be
    physically misleading for pocket characterization.
    """
    from arpes.physics.bz import Lattice3D
    from arpes.physics.bz_overlay import project_hs_points

    meta = (raw_data or {}).get("metadata", {}) or {}
    lat_dict = meta.get("fs_lattice") or {}
    if not lat_dict:
        msg = (
            "No MP lattice: fetch MP symmetry before "
            "displaying the crystal BZ."
        )
        canvas.ax.text(
            0.02, 0.02, msg,
            transform=canvas.ax.transAxes,
            color="#ffcc66", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1a1a1a",
                      edgecolor="#ffcc66", alpha=0.75),
            zorder=8,
        )
        return
    lat = Lattice3D(
        a=float(lat_dict.get("a", p.a_lattice)),
        b=float(lat_dict.get("b", p.b_lattice)),
        c=float(lat_dict.get("c", 1.0) or 1.0),
        alpha_deg=float(lat_dict.get("alpha_deg", 90.0)),
        beta_deg=float(lat_dict.get("beta_deg", 90.0)),
        gamma_deg=float(lat_dict.get("gamma_deg", 90.0)),
        bravais=str(lat_dict.get("bravais", "tetragonal")),
        space_group=str(lat_dict.get("space_group", "")),
        mp_id=str(lat_dict.get("mp_id", p.mp_id)),
    )

    plane = p.kz_plane if p.kz_plane in ("Gamma", "Z") else "Gamma"
    proj, poly = project_hs_points(
        lat,
        plane=plane,
        phi_c_deg=float(p.phi_c_deg),
        gamma_kx=0.0, gamma_ky=0.0,
    )
    if p.overlay_bz_crystal:
        poly_plot = canvas.to_plot_points(poly) if hasattr(canvas, "to_plot_points") else poly
        line, = canvas.ax.plot(
            poly_plot[:, 0], poly_plot[:, 1],
            color="orange", lw=1.4, ls="-", alpha=0.85,
        )
        canvas._overlay_artists.append(line)
    if p.overlay_hs_crystal:
        for pt in proj:
            xy = canvas.to_plot_points([[pt.kx, pt.ky]])[0] if hasattr(canvas, "to_plot_points") else (pt.kx, pt.ky)
            scat = canvas.ax.scatter(
                [xy[0]], [xy[1]], c=pt.color or "orange",
                s=45, zorder=6, edgecolors="black", linewidths=0.5,
            )
            ann = canvas.ax.annotate(
                pt.label, (xy[0], xy[1]), xytext=(5, 5),
                textcoords="offset points",
                color="orange", fontsize=10, fontweight="bold",
            )
            canvas._overlay_artists.extend([scat, ann])
