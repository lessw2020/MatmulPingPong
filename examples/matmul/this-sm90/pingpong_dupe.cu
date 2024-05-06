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


}
