avscuda_sources = [
  'AvsCUDA.cpp',
]

avscuda_headers = [
  'AvsCUDA.h',
  'AvsCUDAversion.h',
]

subdir('filters')

shared_module('AvsCUDA',
  avscuda_sources,
  include_directories : include_directories('..', '../KUtil', '../common'),
  dependencies : [avisynth_dep, cuda_dep, threads_dep, common_dep, avscuda_filters_dep],
  cpp_args : cpp_args,
  link_args : link_args,
  name_prefix : '',
  install : true,
  install_dir : get_option('libdir')
) 