#!/usr/bin/env bash
# Downloads all LESS datasets from Zenodo into data/.
# Usage: bash download_data.sh [poke_primitive|big|multi|handheld|all]
# Default: all

set -euo pipefail

TARGET="${1:-all}"

download_and_extract() {
    local name="$1"
    local url="$2"
    local zip="$3"
    local dest="$4"

    echo "==> Downloading $name..."
    mkdir -p "$dest"
    wget "$url" -O "$zip"
    unzip "$zip" -d "_tmp_${name}/"
    rm -f "$zip"
    local rsync_rc=0
    rsync -a --ignore-existing "_tmp_${name}/"*/ "$dest/" || rsync_rc=$?
    [[ $rsync_rc -eq 0 || $rsync_rc -eq 24 ]] || return $rsync_rc
    rm -rf "_tmp_${name}/"
    echo "==> $name saved to $dest"
}

if [[ "$TARGET" == "poke_primitive" || "$TARGET" == "all" ]]; then
    download_and_extract poke \
        "https://zenodo.org/records/20367204/files/dataset_poke.zip?download=1" \
        dataset_poke.zip \
        data/data_poke_primitive

    download_and_extract primitive \
        "https://zenodo.org/records/20367198/files/dataset_primitive.zip?download=1" \
        dataset_primitive.zip \
        data/data_poke_primitive
fi

if [[ "$TARGET" == "big" || "$TARGET" == "all" ]]; then
    download_and_extract big \
        "https://zenodo.org/records/20367501/files/dataset_poke_big.zip?download=1" \
        dataset_poke_big.zip \
        data/data_poke_big
fi

if [[ "$TARGET" == "multi" || "$TARGET" == "all" ]]; then
    download_and_extract multi \
        "https://zenodo.org/records/20367501/files/dataset_poke_multi.zip?download=1" \
        dataset_poke_multi.zip \
        data/data_poke_multi
fi

if [[ "$TARGET" == "handheld" || "$TARGET" == "all" ]]; then
    download_and_extract handheld \
        "https://zenodo.org/records/20367501/files/dataset_handheld.zip?download=1" \
        dataset_handheld.zip \
        data/data_handheld
fi

echo "Done."
