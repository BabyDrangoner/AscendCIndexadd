/*
 * IndexAdd tiling.
 *
 * Tensor is decomposed as [outer, axis, inner] around dim. dtype_mode:
 * 1=int8, 2=float16, 3=float32, 4=int32, 5=bfloat16.
 */
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(IndexAddTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, outer)
  TILING_DATA_FIELD_DEF(uint32_t, axis)
  TILING_DATA_FIELD_DEF(uint32_t, inner)
  TILING_DATA_FIELD_DEF(uint32_t, dim)
  TILING_DATA_FIELD_DEF(uint32_t, index_len)
  TILING_DATA_FIELD_DEF(uint32_t, index_dtype_size)
  TILING_DATA_FIELD_DEF(uint32_t, dtype_size)
  TILING_DATA_FIELD_DEF(uint32_t, dtype_mode)
  TILING_DATA_FIELD_DEF(uint32_t, ndim)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)

}  // namespace optiling
