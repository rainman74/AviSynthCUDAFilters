project('AviSynthCUDAFilters', 'cpp', 'cuda',
  version : '1.0.0',
  default_options : [
    'cpp_std=c++17',
    'warning_level=1',
    'buildtype=release'
  ]
)

# 依存関係の確認
avisynth_dep = dependency('avisynth', required : true)
cuda_dep = dependency('cuda', version : '>=11.0', required : true)
threads_dep = dependency('threads')

# ビルド設定
cuda_args = [ '-DENABLE_CUDA=1' ]
cpp_args = [ '-DENABLE_CUDA=1' ]
link_args = []
if host_machine.system() == 'linux'
  cpp_args += [ '-std=c++17',
    '-fno-operator-names',
    '-Wno-unused-variable',
    '-Wno-unused-function',
    '-Wno-unused-value',
    '-Wno-unused-label',
    '-Wno-unused-parameter',
    '-Wno-unused-type',
    '-Wno-unknown-pragmas',
    '-Wno-unused-but-set-variable',
    '-Wno-sign-conversion'
  ]
  cuda_args += [ '-std=c++17',
    '-Wno-deprecated-gpu-targets',
    '-allow-unsupported-compiler',
    '-Xcompiler=-Wno-unused-variable',
    '-Xcompiler=-Wno-unused-function',
    '-Xcompiler=-Wno-unused-value',
    '-Xcompiler=-Wno-unused-label',
    '-Xcompiler=-Wno-unused-parameter',
    '-Xcompiler=-Wno-unused-type',
    '-Xcompiler=-Wno-unknown-pragmas',
    '-Xcompiler=-Wno-unused-but-set-variable',
    '-Xcompiler=-Wno-sign-conversion',
    '-Xcompiler=-Wno-maybe-uninitialized',
    '-diag-suppress=177',  # 未使用変数の警告
    '-diag-suppress=550',  # 未初期化変数の警告
    '-diag-suppress=68',   # 符号変換の警告
    '-Xcompiler=-Wno-restrict'  # restrict修飾子の警告
  ]
endif

# SIMD 拡張命令セット用のコンパイルフラグ
sse2_args = cpp_args + ['-msse2']
ssse3_args = cpp_args + ['-mssse3']
sse41_args = cpp_args + ['-msse4.1']
avx_args = cpp_args + ['-mavx', '-mpopcnt']
avx2_args = cpp_args + ['-mavx2', '-mfma', '-mpopcnt', '-mbmi', '-mbmi2']
avx512_args = cpp_args + ['-mavx512f', '-mavx512bw', '-mpopcnt', '-mbmi', '-mbmi2']

# サブディレクトリのビルド設定
subdir('common')
subdir('KUtil')
subdir('KFM')
subdir('NNEDI3')
subdir('masktools')
subdir('KTGMC')
subdir('AvsCUDA')