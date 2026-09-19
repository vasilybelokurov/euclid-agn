#!/usr/bin/env bash
# Build rvspecfit templates for the Euclid NISP red grism from the PHOENIX R=10000 grid.
#
#   source ~/Work/venvs/.venv/bin/activate; bash scripts/rvspecfit_nisp_setup.sh
#
# Products land in ~/data/euclid/rvspecfit/ (config name "nisp_red").  Wavelengths stay in
# vacuum (no --air): Euclid is in space.  NISP LSF for a point source: sigma 13.7 A -> FWHM
# 32.3 A, so R(lambda) = lambda/32.3, given to --resol_func.  Template step 6.7 A = half a
# NISP pixel (13.4 A), range padded beyond the 11900-19002 A archive grid.
set -euo pipefail
PHX=${SPECTRAL_TEMPLATES:-~/data/spectral_templates}/phoenix
OUT=~/data/euclid/rvspecfit
mkdir -p "$OUT"
for z in -0.0 -0.5 -1.0 -2.0; do
  [ -d "$PHX/Z$z" ] || unzip -o -q "$PHX/PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z$z.zip" -d "$PHX/Z$z"
done
# one alpha-enhanced bundle so [alpha/M] spans a range: rvspecfit's CCF builder normalises every grid
# parameter to [0, 1] and asserts on a zero-range parameter, and the parameter count must match the DB
A="PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z-1.0.Alpha=+0.40.zip"
[ -f "$PHX/$A" ] || curl -sS -L -C - -o "$PHX/$A" "https://phoenix.astro.physik.uni-goettingen.de/data/MedResFITS/R10000FITS/$A"
[ -d "$PHX/Z-1.0.Alpha=+0.40" ] || unzip -o -q "$PHX/$A" -d "$PHX/Z-1.0.Alpha=+0.40"
# wavelength vector of the AWAV-LOG grid (identical for every file), as the FITS array rvspecfit expects
python - <<'PY'
import glob, os
import numpy as np
from astropy.io import fits
from pathlib import Path
phx = Path(os.environ.get("SPECTRAL_TEMPLATES", "~/data/spectral_templates")).expanduser() / "phoenix"
f = sorted(phx.glob("Z-0.0/lte*.fits"))[0]
h = fits.getheader(f)
lam = np.exp(h["CRVAL1"] + (np.arange(h["NAXIS1"]) + 1 - h.get("CRPIX1", 1.0)) * h["CDELT1"])
fits.PrimaryHDU(lam.astype(np.float64)).writeto(phx / "WAVE_PHOENIX_R10000.fits", overwrite=True)
print("wavefile", lam[0], lam[-1], lam.size)
PY
rm -f "$OUT/files.db"; rvs_read_grid --prefix "$PHX/" --glob_mask "Z*/lte*.fits" --templdb "$OUT/files.db"  # rvspecfit concatenates prefix+mask: trailing slash required
rvs_make_interpol --setup nisp_red --lambda0 11500 --lambda1 19500 --resol_func "x/32.3" --step 6.7 \
  --templdb "$OUT/files.db" --templprefix "$PHX" --wavefile "$PHX/WAVE_PHOENIX_R10000.fits" \
  --oprefix "$OUT/templ_data" --nthreads 8
rvs_make_nd --prefix "$OUT/templ_data" --setup nisp_red
rvs_make_ccf --prefix "$OUT/templ_data" --oprefix "$OUT/templ_data" --setup nisp_red \
  --lambda0 11500 --lambda1 19500 --step 6.7 --every 4 --nthreads 8
echo "rvspecfit nisp_red configuration built in $OUT/templ_data"
ls -la "$OUT/templ_data" | head
