#!/bin/sh

USER=$(git config --get user.name)
AVISYNTHP_VER=3.7.5

# nvccが見つからなければ、/usr/local/cuda/binをPATHに追加
if ! command -v nvcc &> /dev/null; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

wget https://github.com/AviSynth/AviSynthPlus/archive/refs/tags/v${AVISYNTHP_VER}.tar.gz && \
tar -xzf v${AVISYNTHP_VER}.tar.gz && \
mv AviSynthPlus-${AVISYNTHP_VER} AviSynthPlus && \
cd AviSynthPlus && \
mkdir avisynth-build && \
cd avisynth-build && \
cmake ../ -G Ninja -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
ninja && \
    sudo checkinstall --maintainer="${USER}" --pkgname=avisynth --pkgversion="${AVISYNTHP_VER}" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
    --strip=yes --stripso=yes --addso=yes --fstrans=no --default ninja install