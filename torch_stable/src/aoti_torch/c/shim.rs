// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h
use super::macros::*;

// Keep the order the same as the original file.
unsafe extern "C" {

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L56
    //
    pub unsafe fn aoti_torch_device_type_cpu() -> i32;
    pub unsafe fn aoti_torch_device_type_cuda() -> i32;
    pub unsafe fn aoti_torch_device_type_meta() -> i32;
    pub unsafe fn aoti_torch_device_type_xpu() -> i32;
    pub unsafe fn aoti_torch_device_type_mps() -> i32;
    pub unsafe fn aoti_torch_device_type_privateuse1() -> i32;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L63
    pub unsafe fn aoti_torch_dtype_float8_e5m2() -> i32;
    pub unsafe fn aoti_torch_dtype_float8_e4m3fn() -> i32;
    pub unsafe fn aoti_torch_dtype_float8_e5m2fnuz() -> i32;
    pub unsafe fn aoti_torch_dtype_float8_e4m3fnuz() -> i32;
    // #if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0
    // pub unsafe fn aoti_torch_dtype_float8_e8m0fnu() -> i32;
    // pub unsafe fn aoti_torch_dtype_float4_e2m1fn_x2() -> i32;
    // #endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0
    pub unsafe fn aoti_torch_dtype_bfloat16() -> i32;
    pub unsafe fn aoti_torch_dtype_float16() -> i32;
    pub unsafe fn aoti_torch_dtype_float32() -> i32;
    pub unsafe fn aoti_torch_dtype_float64() -> i32;
    pub unsafe fn aoti_torch_dtype_uint8() -> i32;
    pub unsafe fn aoti_torch_dtype_uint16() -> i32;
    pub unsafe fn aoti_torch_dtype_uint32() -> i32;
    pub unsafe fn aoti_torch_dtype_uint64() -> i32;
    pub unsafe fn aoti_torch_dtype_int8() -> i32;
    pub unsafe fn aoti_torch_dtype_int16() -> i32;
    pub unsafe fn aoti_torch_dtype_int32() -> i32;
    pub unsafe fn aoti_torch_dtype_int64() -> i32;
    pub unsafe fn aoti_torch_dtype_bool() -> i32;
    pub unsafe fn aoti_torch_dtype_complex32() -> i32;
    pub unsafe fn aoti_torch_dtype_complex64() -> i32;
    pub unsafe fn aoti_torch_dtype_complex128() -> i32;
    pub unsafe fn aoti_torch_dtype_element_size(dtype: i32) -> usize;

    // Layouts
    //
    pub unsafe fn aoti_torch_layout_strided() -> i32;
    pub unsafe fn aoti_torch_layout_sparse_coo() -> i32;
    pub unsafe fn aoti_torch_layout_sparse_csr() -> i32;
    pub unsafe fn aoti_torch_layout_sparse_csc() -> i32;
    pub unsafe fn aoti_torch_layout_sparse_bsr() -> i32;
    pub unsafe fn aoti_torch_layout_sparse_bsc() -> i32;
    pub unsafe fn aoti_torch_layout__mkldnn() -> i32;
    pub unsafe fn aoti_torch_layout_jagged() -> i32;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L97C1-L102C1
    pub unsafe fn aoti_torch_memory_format_contiguous_format() -> i32;
    pub unsafe fn aoti_torch_memory_format_channels_last() -> i32;
    pub unsafe fn aoti_torch_memory_format_channels_last_3d() -> i32;
    pub unsafe fn aoti_torch_memory_format_preserve_format() -> i32;

    // Get TORCH_ABI_VERSION of the built libtorch.so
    pub unsafe fn aoti_torch_abi_version() -> u64;
    // Not in my version :< Need to update my driver to use the latest libtorch :/
    // Updated my driver, also needed that to parse the device string... and the stable api only exists from 2.10 really anyway.

    // Conversions
    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L106

    pub unsafe fn aoti_torch_item_float32(
        tensor: AtenTensorHandle,
        ret_value: *mut f32,
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_item_float64(
        tensor: AtenTensorHandle,
        ret_value: *mut f64,
    ) -> AOTITorchError;

    // Scalar to single element tensor
    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L137
    pub unsafe fn aoti_torch_scalar_to_tensor_float32(
        value: f32,
        ret_new_tensor: &mut AtenTensorHandle,
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_scalar_to_tensor_float64(
        value: f64,
        ret_new_tensor: &mut AtenTensorHandle,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L180
    //
    pub unsafe fn aoti_torch_delete_tensor_object(tensor: AtenTensorHandle) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/e0347a69ec5546b16b89f3665be60a0932905f19/torch/csrc/inductor/aoti_torch/c/shim.h#L212

    // Get the nbytes of the underlying storage
    pub unsafe fn aoti_torch_get_storage_size(
        tensor: AtenTensorHandle,
        ret_size: *mut i64,
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_get_dim(tensor: AtenTensorHandle, ret_dim: *mut i64)
    -> AOTITorchError;

    pub unsafe fn aoti_torch_get_numel(
        tensor: AtenTensorHandle,
        ret_numel: *mut i64,
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_get_storage_numel(
        tensor: AtenTensorHandle,
        ret_numel: *mut i64,
    ) -> AOTITorchError;

    /// Returns a borrowed reference
    pub unsafe fn aoti_torch_get_sizes(
        tensor: AtenTensorHandle,
        ret_sizes: &mut *const i64, // int64_t** ret_sizes // returns borrowed reference
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_get_size(
        tensor: AtenTensorHandle,
        d: i64,
        ret_size: *mut i64,
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_get_strides(
        tensor: AtenTensorHandle,
        ret_strides: &mut *const i64, // int64_t** ret_strides // returns borrowed reference
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_get_stride(
        tensor: AtenTensorHandle,
        d: i64,
        ret_stride: *mut i64,
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_get_dtype(
        tensor: AtenTensorHandle,
        ret_dtype: *mut i32,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L244

    pub unsafe fn aoti_torch_get_device_type(
        tensor: AtenTensorHandle,
        ret_device_type: &mut i32,
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_get_device_index(
        tensor: AtenTensorHandle,
        ret_device_type: &mut i32,
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_get_layout(
        tensor: AtenTensorHandle,
        ret_layout: *mut i32,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/e0347a69ec5546b16b89f3665be60a0932905f19/torch/csrc/inductor/aoti_torch/c/shim.h#L256C1-L258C76
    pub unsafe fn aoti_torch_is_contiguous(
        tensor: AtenTensorHandle,
        ret_is_contiguous: &mut bool,
    ) -> AOTITorchError;

    pub unsafe fn aoti_torch_is_defined(
        tensor: AtenTensorHandle,
        ret_is_contiguous: &mut bool,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L289
    // New tensor object, returned through *out, caller has to clear it.
    pub unsafe fn aoti_torch_empty_strided(
        ndim: i64,
        sizes_ptr: *const i64,
        strides_ptr: *const i64,
        dtype: i32,
        device_type: i32,
        device_index: i32,
        ret_new_tensor: &mut AtenTensorHandle,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L345
    pub unsafe fn aoti_torch_new_uninitialized_tensor(ret: &mut AtenTensorHandle)
    -> AOTITorchError;
}

// https://github.com/pytorch/pytorch/blob/3848e11d554a7f49925b593c40b8be0b86ac6b3f/docs/source/notes/libtorch_stable_abi.md#stableivalue-conversions

// https://github.com/pytorch/pytorch/blob/3848e11d554a7f49925b593c40b8be0b86ac6b3f/torch/csrc/inductor/aoti_torch/c/shim.h#L485-L486
// pub type StableIValue = u64;
#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct StableIValue(pub u64);
