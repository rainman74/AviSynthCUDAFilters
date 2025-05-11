common_sources = [
  'DebugWriter.cpp',
  'DeviceLocalData.cpp',
  'Copy.cu',
  'rgy_filesystem.cpp',
  'rgy_util.cpp',
  'rgy_simd.cpp',
  'rgy_event.cpp'
]

common_headers = [
  'CommonFunctions.h',
  'Copy.h',
  'DebugWriter.h',
  'DeviceLocalData.h',
  'EnvCommon.h',
  'Frame.h',
  'KMV.h',
  'KVersion.h',
  'ReduceKernel.cuh',
  'VectorFunctions.cuh',
]

common_lib = static_library('common',
  common_sources,
  include_directories : include_directories('..', '../KUtil'),
  dependencies : [avisynth_dep, cuda_dep, threads_dep],
  cpp_args : cpp_args,
  cuda_args : cuda_args,
)

common_dep = declare_dependency(
  link_with : common_lib,
  include_directories : include_directories('.')
) 