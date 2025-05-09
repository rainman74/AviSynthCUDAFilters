ktgmc_sources = [
  'MV.cpp',
  'Misc.cpp',
  'Kernel.cu',
  'MVKernel.cu',
]

ktgmc_headers = [
  'CudaDebug.h',
  'CudaKernelBase.h',
  'GenericImageFunctions.cuh',
  'KTGMCversion.h',
  'MVKernel.h',
  'Misc.h',
]

shared_module('KTGMC',
  ktgmc_sources,
  include_directories : include_directories('..', '../KUtil', '../common'),
  dependencies : [avisynth_dep, cuda_dep, threads_dep, common_dep],
  cpp_args : cpp_args,
  cuda_args : cuda_args,
  link_args : link_args,
  name_prefix : '',
  install : true,
  install_dir : get_option('libdir')
) 