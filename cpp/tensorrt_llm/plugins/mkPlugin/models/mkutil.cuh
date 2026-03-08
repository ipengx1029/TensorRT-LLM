#pragma once
#include "base.h"
#include "kittens.cuh"

namespace mk {
template<kittens::ducks::gl::all GL, bool only_cols=false>
GL convert2gl(MKTensor &tensor) {
    int dims = tensor.dim->nbDims;
    if (dims > 4) {
        throw std::runtime_error("Expected Tensor.ndim <= 4");
    }
    std::array<int, 4> arr = {1, 1, 1, 1};
    if constexpr(only_cols) {
        arr[3] = tensor.dim->d[dims - 1];
    } else {
        for (size_t i = 0; i < dims; ++i) {
            arr[4 - dims + i] = tensor.dim->d[i];
        }
    }
    uint64_t data_ptr = (uint64_t)(tensor.ptr);
    // Create GL object using make_gl
    return kittens::make_gl<GL>(data_ptr, arr[0], arr[1], arr[2], arr[3]);
}
template<kittens::ducks::gl::all GL, bool only_cols=false>
GL convert2gl(MKGLTensor &tensor) {
    int dims = tensor.dim.nbDims;
    if (dims > 4) {
        throw std::runtime_error("Expected Tensor.ndim <= 4");
    }
    std::array<int, 4> arr = {1, 1, 1, 1};
    if constexpr(only_cols) {
        arr[3] = tensor.dim.d[dims - 1];
    } else {
        for (size_t i = 0; i < dims; ++i) {
            arr[4 - dims + i] = tensor.dim.d[i];
        }
    }
    uint64_t data_ptr = (uint64_t)(tensor.ptr);
    // Create GL object using make_gl
    return kittens::make_gl<GL>(data_ptr, arr[0], arr[1], arr[2], arr[3]);
}
};
