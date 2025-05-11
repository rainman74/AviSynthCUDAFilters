# AviSynthCUDAFilters

このプロジェクトはmesonビルドシステムを使用してLinux環境でビルドすることができます。

## 注意

Windows版と比べ、以下の機能制限があります。

- nnediのCPU版のアセンブラコードは動作しません。

## 想定動作環境

- Linux x64
- NVIDIA製GPUを搭載し、適切にドライバを導入した環境

上記以外の環境は対象外です。

# Ubuntu 22.04/24.04 インストール方法

[こちら](https://github.com/rigaya/AviSynthCUDAFilters/releases)から

- CUDAを有効にしたAvisynthPlus
- AvisynthCUDAFilters

の2つをダウンロードしてインストールします。

```bash
sudo dpkg -i ./avisynth_xxx.deb
sudo dpkg -i ./avisynthcudafilters_xxx.deb
```

---

# Linux ビルド手順

ビルドにあたっては、

- CUDAのインストール可能な環境であること
- インストールされたCUDAに対応するgccコンパイラがインストールされていること

であることが必要です。それ以外の環境でのビルドは困難と思われます。

## 依存関係のインストール

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install git cmake meson ninja-build build-essential pkg-config
```

### CUDAのインストール

ご自身の環境に合わせ、[CUDA](https://developer.nvidia.com/cuda-downloads)をインストールします。

インストール後、```nvcc```コマンドが実行可能か確認します。実行できない場合、下記でCUDAのディレクトリをパスに加えます。

```bash
export PATH=/usr/local/cuda/bin:${PATH}
```

### AviSynth+のビルドとインストール

CUDAを有効にしてビルドします。

```bash
(git clone https://github.com/AviSynth/AviSynthPlus.git \
  && cd AviSynthPlus && mkdir build && cd build \
  && cmake -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release .. \
  && make -j$(nproc) \
  && sudo make install)
```

## AviSynthCUDAFiltersのビルド

AviSynth+をインストール後、下記を実行します。

```bash
(git clone https://github.com/rigaya/AviSynthCUDAFilters.git \
  && mkdir build && cd build && meson setup --buildtype release .. \
  && ninja \
  && sudo ninja install)
```

デフォルトでは、プラグインは```/usr/local/lib/avisynth```にインストールされます。
