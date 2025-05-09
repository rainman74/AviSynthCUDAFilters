kutil_sources = [
  'KUtil.cpp',
]

kutil_headers = [
  'KUtilversion.h',
]

shared_module('KUtil',
  kutil_sources,
  include_directories : include_directories('..', '../common', '../KUtil'),
  dependencies : [avisynth_dep, threads_dep],
  cpp_args : cpp_args,
  link_args : link_args,
  name_prefix : '',
  install : true,
  install_dir : get_option('libdir')
) 