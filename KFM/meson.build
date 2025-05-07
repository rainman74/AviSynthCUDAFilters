kfm_sources = [
  'KFM.cpp',
  'CombingAnalyze.cu',
  'Deblock.cu',
  'DecombeUCF.cu',
  'KDeband.cu',
  'KFMFilterBase.cu',
  'KFMKernel.cu',
  'MergeStatic.cu',
  'TextOut.cu',
]

kfm_headers = [
  'KFM.h',
  'KFMFilterBase.cuh',
  'KFMversion.h',
  'SIMDSupport.hpp',
  'TextOut.h',
]

# AVX用のソースファイル
avx2_sources = ['DeblockAVX.cpp']

# AVXファイル用のライブラリ
avx2_lib = static_library('KFM_avx',
  avx2_sources,
  include_directories : include_directories('..', '../../KUtil', '../../common'),
  dependencies : [avisynth_dep, cuda_dep, threads_dep, common_dep],
  cpp_args : avx2_args + ['-mavx2', '-mfma'],
  cuda_args : cuda_args,
  pic : true,
)

shared_module('KFM',
  kfm_sources,
  include_directories : include_directories('..', '../KUtil', '../common'),
  dependencies : [avisynth_dep, cuda_dep, threads_dep, common_dep],
  link_with : [avx2_lib],
  cpp_args : cpp_args + ['-mavx2', '-mfma'],
  cuda_args : cuda_args,
  link_args : link_args,
  name_prefix : '',
  install : true,
  install_dir : get_option('libdir') / 'avisynth'
)
