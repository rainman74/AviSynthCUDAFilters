# AviSynthCUDAFilters - Linux ビルド手順

このプロジェクトはmesonビルドシステムを使用してLinux環境でビルドすることができます。

## 必要条件

- CUDA 11.0以上
- AviSynth+
- Meson
- Ninja
- g++ または clang++
- pkg-config

## 依存関係のインストール

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install meson ninja-build build-essential pkg-config
```

### AviSynth+のインストール
AviSynth+はpkg-configを通して検出される必要があります。[AviSynth+のインストール手順](https://github.com/AviSynth/AviSynthPlus)を参照してください。

### CUDAのインストール
NVIDIAの公式サイトから[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)をダウンロードしてインストールしてください。CUDA 11.0以上が必要です。

## ビルド手順

```bash
mkdir build && cd build && meson setup ..
ninja

# インストール (オプション)
sudo ninja install
```

デフォルトでは、プラグインは`/usr/local/lib/avisynth`にインストールされます。

## トラブルシューティング

### CUDAが見つからない場合
CUDA Toolkitのインストールパスが`pkg-config`で見つけられない場合は、以下のように環境変数を設定してください：

```bash
export PKG_CONFIG_PATH=/usr/local/cuda/lib64/pkgconfig:$PKG_CONFIG_PATH
```

### AviSynthが見つからない場合
AviSynth+がpkg-configを通して見つけられない場合は、インストールパスを確認し、必要に応じて`PKG_CONFIG_PATH`環境変数を設定してください。 