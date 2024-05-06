#include "common.h"
#include "reference.h"

const int testM = 5120;
const int testN = 4096;
const int testK = 2048;

static constexpr int cluster_M = 2;
static constexpr int cluster_N = 1;
static constexpr int wg_number = 3;

static constexpr int blockM = 128;
static constexpr int blockN = 128;
static constexpr int blockK = 64;
static constexpr int stages = 7;

namespace utils {

using tmaDescriptor = CUtensorMap;

template <class T>
inline CUtensorMapDataType to_CUtensorMapDataType() {
    if constexpr (std::is_same<T, int8_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    } else if constexpr (std::is_same<T, uint16_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    } else if constexpr (std::is_same<T, uint32_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    } else if constexpr (std::is_same<T, uint64_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT64;
    } else if constexpr (std::is_same<T, int32_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_INT32;
    } else if constexpr (std::is_same<T, int64_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_INT64;
    } else if constexpr (std::is_same<T, half_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if constexpr (std::is_same<T, float>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else if constexpr (std::is_same<T, double>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    } else if constexpr (std::is_same<T, bfloat16>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if constexpr (std::is_same<T, tfloat32_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;
    } else {
        static_assert(sizeof(T) < 0, "Unknown TMA Format!");
    }
}

enum class SmemSwizzleBits : uint8_t {
    DISABLE = 0,
    B32 = 1,
    B64 = 2,
    B128 = 3,
};

template <int B, int M, int S>
HOST_DEVICE constexpr SmemSwizzleBits get_tma_swizzle_bits(Swizzle<B, M, S>) {
    if constexpr (M==4) {
        switch (B) {
            default:
            static assert(0 <=B && B <=3, "expected B = 0,1,2 or 3 when M==4. Unsupported layout swizzle.");
            case 3:
                return SmemSwizzleBits::B128;
            case 2:
                return SmemSwizzleBits::B64;
            case 1:
                return SmemSwizzleBits::B32;
            case 0:
                return SmemSwizzleBits::DISABLE;


        }

    }
} else {
    static_assert(M < 0, "unsupported layout swizzle.");
}
}


inline CUtensorMapSwizzle to_CUtensorMapSwizzle(SmemSwizzleBits const& t) {
  switch (t) {
    default:
      assert(false && "Unknown SmemSwizzleBits!");
    case SmemSwizzleBits::DISABLE:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case SmemSwizzleBits::B32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case SmemSwizzleBits::B64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case SmemSwizzleBits::B128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
  }
}


}// minor dims move faster than major
template <int BlockMajorSize, int BlockMinorSize, int B, int M, int S>
TmaDescriptor make_tma_copy_desc(DType* gmem_ptr, int shape_major,
                                int shape_minor,
                                Swizzle<B,M,S> const& swizzle,
                                uint32_t num_multicast) {
void* gmem_address = (void*)gmem_ptr;
uint64_t gmem_prob_shape[5] = {(uint64_t) shape_minor, (uint64_t)shape_major, 1,1,1};
uint64)t gmem_prob_stride[5] = {sizeof(DType), sizeof(DType) * shape_minor, 0, 0,0};




}
