# AviSynth CUDA Filters

This is a CUDA implementation filter plugin for [AviSynthNeo](https://github.com/rainman74/AviSynthNeo).  
The following projects are included:

- KTGMC (CUDA version of QTGMC)  
- KNNEDI3 (CUDA version of NNEDI3)  
- KFM (originally implemented filter)  
- AvsCUDA (CUDA-enabled version of internal AviSynth filters)  
- GRunT (Neo-compatible version of GrunT)  

## Integrated External Projects

This repository includes the complete source code of the following external projects:

- **NNEDI3** (from [rainman74/NNEDI3](https://github.com/rainman74/NNEDI3)) - Located in `NNEDI3/`
- **masktools** (from [rainman74/masktools](https://github.com/rainman74/masktools)) - Located in `masktools/`

These projects have been integrated directly (not as submodules) to ensure full source code availability and compatibility with the Visual Studio build system.

# Usage

[AviSynthNeo](https://github.com/rainman74/AviSynthNeo) is required.  
For filter usage, see the [KTGMC documentation](https://github.com/rainman74/AviSynthCUDAFilters/wiki/KTGMC) and [KFM documentation](https://github.com/rainman74/AviSynthCUDAFilters/wiki/KFM).  
For deinterlacing-related functionality, most features can be accessed using [KFMDeint](https://github.com/rainman74/AviSynthCUDAFilters/wiki/KFMDeint).

# License
- KTGMC: GPL  
- KNNEDI3: GPL  
- KFM: **MIT**  
- AvsCUDA: GPL  
- GRunT: GPL  

## External Project Licenses
- **NNEDI3** (`NNEDI3/`): GPL v2 (see `NNEDI3/gpl.txt`)
- **masktools** (`masktools/`): MIT License (see `masktools/LICENSE`)
