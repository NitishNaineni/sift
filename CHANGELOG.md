# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-10-11

### Added
- Initial release of CUDA SIFT library
- High-performance GPU-accelerated SIFT implementation
- Clean, modular package structure (`cudasift`)
- Comprehensive API with `SiftDetector` class
- Full type annotations throughout
- 44 unit tests comparing against C reference implementation
- Example scripts for basic extraction, batch processing, and visualization
- Comprehensive README with usage examples
- Complete documentation with docstrings
- Legacy `Sift` class for backward compatibility
- Utility functions for CUDA device info and image loading
- Support for batch processing
- Optional intermediate result recording for debugging

### Features
- Fast GPU acceleration using CUDA and Numba
- Matches reference C implementation accuracy
- Flexible parameter configuration
- BT.709 standard grayscale conversion
- CUDA graph optimization for repeated execution
- Buffer overflow detection and warnings
- Easy-to-use Pythonic API

### Package Structure
- `cudasift.core`: Main SIFT detector API
- `cudasift.types`: Data structures and parameter classes  
- `cudasift.kernels`: CUDA kernel implementations
- `cudasift.utils`: Utility functions
- `cudasift.version`: Version information

### Documentation
- Comprehensive README.md
- API documentation in docstrings
- Three example scripts demonstrating usage
- Installation instructions
- Troubleshooting guide

### Testing
- 44 unit tests with full coverage of SIFT pipeline
- Tests compare GPU implementation against C reference
- Test data download script included

[0.1.0]: https://github.com/yourusername/cudasift/releases/tag/v0.1.0
