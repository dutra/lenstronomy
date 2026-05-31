from pathlib import Path

import numpy as np
import pytest

from playground.lenstool_parser import _load_dat_catalog, load_best_par
from playground.rank_active_scaling_galaxies import _parse_reference_header


def _write_catalog(path: Path, header: str, row: str = "1 64.0 -24.0 1 1 0 20 1") -> Path:
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")
    return path


def test_load_dat_catalog_accepts_reference_header_without_space(tmp_path):
    catalog_path = _write_catalog(tmp_path / "catalog_no_space.cat", "#REFERENCE 0")

    df = _load_dat_catalog(catalog_path, par_reference=None, catalog_kind="potfile_galaxies")

    assert int(df.loc[0, "catalog_reference"]) == 0
    assert np.isclose(float(df.loc[0, "ra"]), 64.0)
    assert np.isclose(float(df.loc[0, "dec"]), -24.0)


def test_load_dat_catalog_accepts_reference_header_with_space(tmp_path):
    catalog_path = _write_catalog(tmp_path / "catalog_with_space.cat", "# REFERENCE 0")

    df = _load_dat_catalog(catalog_path, par_reference=None, catalog_kind="potfile_galaxies")

    assert int(df.loc[0, "catalog_reference"]) == 0
    assert np.isclose(float(df.loc[0, "ra"]), 64.0)
    assert np.isclose(float(df.loc[0, "dec"]), -24.0)


@pytest.mark.parametrize("header", ["#REFERENCE 3", "# REFERENCE 3"])
def test_load_dat_catalog_accepts_reference_three_spellings(tmp_path, header):
    catalog_path = _write_catalog(tmp_path / "catalog_ref3.cat", header, row="1 1.0 2.0 1 1 0 20 1")

    df = _load_dat_catalog(
        catalog_path,
        par_reference=(3, 64.0381417, -24.0674722),
        catalog_kind="potfile_galaxies",
    )

    assert int(df.loc[0, "catalog_reference"]) == 3
    assert np.isfinite(float(df.loc[0, "ra"]))
    assert np.isfinite(float(df.loc[0, "dec"]))


def test_load_dat_catalog_produces_identical_values_for_both_header_spellings(tmp_path):
    compact_path = _write_catalog(tmp_path / "compact.cat", "#REFERENCE 3", row="1 1.0 2.0 1 1 0 20 1")
    spaced_path = _write_catalog(tmp_path / "spaced.cat", "# REFERENCE 3", row="1 1.0 2.0 1 1 0 20 1")
    par_reference = (3, 64.0381417, -24.0674722)

    compact_df = _load_dat_catalog(compact_path, par_reference=par_reference, catalog_kind="potfile_galaxies")
    spaced_df = _load_dat_catalog(spaced_path, par_reference=par_reference, catalog_kind="potfile_galaxies")

    assert int(compact_df.loc[0, "catalog_reference"]) == int(spaced_df.loc[0, "catalog_reference"])
    assert np.isclose(float(compact_df.loc[0, "ra"]), float(spaced_df.loc[0, "ra"]))
    assert np.isclose(float(compact_df.loc[0, "dec"]), float(spaced_df.loc[0, "dec"]))


def test_load_dat_catalog_still_rejects_missing_reference_header(tmp_path):
    catalog_path = _write_catalog(tmp_path / "missing_header.cat", "# comment only")

    with pytest.raises(ValueError, match="Missing #REFERENCE header"):
        _load_dat_catalog(catalog_path, par_reference=None, catalog_kind="potfile_galaxies")


@pytest.mark.parametrize("header", ["#REFERENCE 0", "# REFERENCE 0"])
def test_rank_active_scaling_parser_accepts_both_header_spellings(tmp_path, header):
    catalog_path = _write_catalog(tmp_path / "ranking.cat", header)

    parsed_header, rows = _parse_reference_header(catalog_path)

    assert parsed_header.reference == 0
    assert rows == [["1", "64.0", "-24.0", "1", "1", "0", "20", "1"]]


def test_load_best_par_accepts_m0416_catalog_with_spaced_reference_header():
    par_path = Path("playground/M0416_Bergamini22/Bergamini22_MACS0416.par")

    parsed, potentials_df, images_df, potentials_with_priors = load_best_par(par_path)

    assert parsed["potfiles"]
    assert not potentials_df.empty
    assert not images_df.empty
    assert len(potentials_with_priors) > 0
    potfile = parsed["potfiles"][0]
    assert potfile["corekpc"] is None
    assert np.isclose(float(potfile["core_arcsec"]), 0.0001)
    assert potfile["cutkpc_nominal"] is None
    assert np.isclose(float(potfile["cut_arcsec_nominal"]), 25.5)
    first_potential = potentials_with_priors[0]
    assert "core_radius_kpc" in first_potential
    assert "cut_radius_kpc" in first_potential
    assert first_potential["core_radius_kpc"] > 0.0
    assert first_potential["cut_radius_kpc"] > first_potential["core_radius_kpc"]
    assert "core_radius_kpc" in first_potential["priors"]


def test_load_best_par_converts_large_dpie_arcsec_radii_to_kpc(tmp_path):
    par_path = tmp_path / "input.par"
    par_path.write_text(
        "\n".join(
            [
                "runmode",
                "    reference 3 10.0 -20.0",
                "    end",
                "cosmology",
                "    H0 70.0",
                "    omega 0.3",
                "    lambda 0.7",
                "    end",
                "potentiel 1",
                "    profil 81",
                "    x_centre 0.0",
                "    y_centre 0.0",
                "    ellipticite 0.2",
                "    angle_pos 15.0",
                "    core_radius 2.0",
                "    cut_radius 20.0",
                "    v_disp 300.0",
                "    z_lens 0.4",
                "    end",
                "limit 1",
                "    core_radius 1 1.0 3.0 0.1",
                "    cut_radius 1 10.0 30.0 0.1",
                "    end",
                "limit 99",
                "    x_centre 0 0.0",
                "    end",
                "fini",
            ]
        ),
        encoding="utf-8",
    )

    parsed, potentials_df, _images_df, potentials_with_priors = load_best_par(par_path)

    assert not potentials_df.empty
    assert "core_radius_kpc" in potentials_df.columns
    assert "cut_radius_kpc" in potentials_df.columns
    assert float(potentials_df.loc[0, "core_radius_kpc"]) > 0.0
    assert float(potentials_df.loc[0, "cut_radius_kpc"]) > float(potentials_df.loc[0, "core_radius_kpc"])
    priors = potentials_with_priors[0]["priors"]
    assert "core_radius_kpc" in priors
    assert "cut_radius_kpc" in priors
    assert "core_radius" not in priors
    assert "cut_radius" not in priors


def test_load_best_par_raises_when_large_dpie_radius_fields_are_missing(tmp_path):
    par_path = tmp_path / "missing_radius.par"
    par_path.write_text(
        "\n".join(
            [
                "runmode",
                "    reference 3 10.0 -20.0",
                "    end",
                "potentiel 1",
                "    profil 81",
                "    x_centre 0.0",
                "    y_centre 0.0",
                "    ellipticite 0.2",
                "    angle_pos 15.0",
                "    v_disp 300.0",
                "    z_lens 0.4",
                "    end",
                "fini",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="core_radius"):
        load_best_par(par_path)
