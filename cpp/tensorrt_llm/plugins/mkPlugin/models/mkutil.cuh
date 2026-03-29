#pragma once
#include "base.h"
#include "kittens.cuh"

namespace mk {
template<kittens::ducks::gl::all GL, bool only_cols=false>
GL convert2gl(MKTensor &tensor) {
    int dims = tensor.dim->nbDims;
    if (dims > 5) {
        throw std::runtime_error("Expected Tensor.ndim <= 5");
    }
    std::array<int, 4> arr = {1, 1, 1, 1};
    if constexpr(only_cols) {
        arr[3] = tensor.dim->d[dims - 1];
    } else {
        if (dims == 5) {
            arr[0] = tensor.dim->d[0] * tensor.dim->d[1];
            for (size_t i = 2; i < 5; ++i) {
                arr[i - 1] = tensor.dim->d[i];
            }
        } else {
            for (size_t i = 0; i < dims; ++i) {
                arr[4 - dims + i] = tensor.dim->d[i];
            }
        }
    }
    uint64_t data_ptr = (uint64_t)(tensor.ptr);
    // Create GL object using make_gl
    return kittens::make_gl<GL>(data_ptr, arr[0], arr[1], arr[2], arr[3]);
}
template<kittens::ducks::gl::all GL, bool only_cols=false>
GL convert2gl(MKGLTensor &tensor) {
    int dims = tensor.dim.nbDims;
    if (dims > 5) {
        throw std::runtime_error("Expected Tensor.ndim <= 5");
    }
    std::array<int, 4> arr = {1, 1, 1, 1};
    if constexpr(only_cols) {
        arr[3] = tensor.dim.d[dims - 1];
    } else {
        if (dims == 5) {
            arr[0] = tensor.dim.d[0] * tensor.dim.d[1];
            for (size_t i = 2; i < 5; ++i) {
                arr[i - 1] = tensor.dim.d[i];
            }
        } else {
            for (size_t i = 0; i < dims; ++i) {
                arr[4 - dims + i] = tensor.dim.d[i];
            }
        }
    }
    uint64_t data_ptr = (uint64_t)(tensor.ptr);
    // Create GL object using make_gl
    return kittens::make_gl<GL>(data_ptr, arr[0], arr[1], arr[2], arr[3]);
}
template<kittens::ducks::gl::all GL, size_t N, bool only_cols = false>
std::array<GL, N> convert2gl_array(MKTensor* tensor_array) {
    return convert2gl_array_impl<GL, N, only_cols>(std::make_index_sequence<N>{}, tensor_array);
}
template<kittens::ducks::gl::all GL, size_t N, bool only_cols, size_t... I>
std::array<GL, N> convert2gl_array_impl(std::index_sequence<I...>, MKTensor* tensor_array) {
    return std::array<GL, N>{convert2gl<GL, only_cols>(tensor_array[I])...};
}
};