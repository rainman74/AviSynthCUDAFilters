#!/bin/sh

VER=$(git describe --tags)
SHORTVER=$(echo $VER | sed -E 's/-g[0-9a-f]+$//')
#gitコマンドのユーザーを取得
USER=$(git config --get user.name)

mkdir build && \
cd build && \
meson setup --buildtype release .. && \
ninja && \
    sudo checkinstall --pkgname=AviSynthCUDAFilters --maintainer="${USER}" --pkgversion="${SHORTVER}" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
    --strip=yes --stripso=yes --addso=yes --fstrans=no --default ninja install