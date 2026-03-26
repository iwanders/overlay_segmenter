// https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html
//
// Docs aren't great, the the stable c++ api is well documented:
// https://github.com/pytorch/pytorch/tree/main/torch/csrc/stable
//
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h#L21

// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/macros.h#L46

// Hmm, the stable c++ interface actually goes through the call inteface; https://github.com/pytorch/pytorch/blob/e3d4dcc4adddff454ceeef71b3d735acc5e23344/torch/csrc/stable/ops.h#L625
//
//
// Bunch of tests here: https://github.com/pytorch/pytorch/tree/1c0fd99bc15e998eab3cee79588544a033f0e4df/test/cpp_extensions/libtorch_agn_2_10_extension/csrc
// empty is here: https://github.com/pytorch/pytorch/blob/1c0fd99bc15e998eab3cee79588544a033f0e4df/test/cpp_extensions/libtorch_agn_2_10_extension/csrc/my_empty.cpp#L10-L18
// Through; https://github.com/pytorch/pytorch/blob/1c0fd99bc15e998eab3cee79588544a033f0e4df/test/cpp_extensions/libtorch_agn_2_10_extension/csrc/my_empty.cpp#L10-L18

// Maybe the problem is with my torch version?

#[repr(C)]
struct AtenTensorOpaque {
    _private: [u8; 0],
}
type AtenTensorHandle = *mut AtenTensorOpaque;
type AOTITorchError = i32;

unsafe extern "C" {
    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L56
    //
    fn aoti_torch_device_type_cpu() -> i32;
    fn aoti_torch_device_type_cuda() -> i32;
    fn aoti_torch_device_type_meta() -> i32;
    fn aoti_torch_device_type_xpu() -> i32;
    fn aoti_torch_device_type_mps() -> i32;
    fn aoti_torch_device_type_privateuse1() -> i32;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L63
    fn aoti_torch_dtype_float8_e5m2() -> i32;
    fn aoti_torch_dtype_float8_e4m3fn() -> i32;
    fn aoti_torch_dtype_float8_e5m2fnuz() -> i32;
    fn aoti_torch_dtype_float8_e4m3fnuz() -> i32;
    // #if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0
    // fn aoti_torch_dtype_float8_e8m0fnu() -> i32;
    // fn aoti_torch_dtype_float4_e2m1fn_x2() -> i32;
    // #endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0
    fn aoti_torch_dtype_bfloat16() -> i32;
    fn aoti_torch_dtype_float16() -> i32;
    fn aoti_torch_dtype_float32() -> i32;
    fn aoti_torch_dtype_float64() -> i32;
    fn aoti_torch_dtype_uint8() -> i32;
    fn aoti_torch_dtype_uint16() -> i32;
    fn aoti_torch_dtype_uint32() -> i32;
    fn aoti_torch_dtype_uint64() -> i32;
    fn aoti_torch_dtype_int8() -> i32;
    fn aoti_torch_dtype_int16() -> i32;
    fn aoti_torch_dtype_int32() -> i32;
    fn aoti_torch_dtype_int64() -> i32;
    fn aoti_torch_dtype_bool() -> i32;
    fn aoti_torch_dtype_complex32() -> i32;
    fn aoti_torch_dtype_complex64() -> i32;
    fn aoti_torch_dtype_complex128() -> i32;
    fn aoti_torch_dtype_element_size(dtype: i32) -> usize;

    // Layouts
    //
    fn aoti_torch_layout_strided() -> i32;
    fn aoti_torch_layout_sparse_coo() -> i32;
    fn aoti_torch_layout_sparse_csr() -> i32;
    fn aoti_torch_layout_sparse_csc() -> i32;
    fn aoti_torch_layout_sparse_bsr() -> i32;
    fn aoti_torch_layout_sparse_bsc() -> i32;
    fn aoti_torch_layout__mkldnn() -> i32;
    fn aoti_torch_layout_jagged() -> i32;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L97C1-L102C1
    fn aoti_torch_memory_format_contiguous_format() -> i32;
    fn aoti_torch_memory_format_channels_last() -> i32;
    fn aoti_torch_memory_format_channels_last_3d() -> i32;
    fn aoti_torch_memory_format_preserve_format() -> i32;

    // Get TORCH_ABI_VERSION of the built libtorch.so
    //fn aoti_torch_abi_version() -> u64;
    // Not in my version :<
    //

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L244

    fn aoti_torch_get_device_type(
        tensor: AtenTensorHandle,
        ret_device_type: &mut i32,
    ) -> AOTITorchError;
    fn aoti_torch_get_device_index(
        tensor: AtenTensorHandle,
        ret_device_type: &mut i32,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L180
    //
    fn aoti_torch_delete_tensor_object(tensor: AtenTensorHandle) -> AOTITorchError;

    fn aoti_torch_new_uninitialized_tensor(ret: &mut AtenTensorHandle) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L137
    fn aoti_torch_scalar_to_tensor_float32(
        value: f32,
        ret_new_tensor: &mut AtenTensorHandle,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/shim.h#L293
    fn aoti_torch_empty_strided(
        ndim: i64,
        sizes_ptr: *const i64,
        strides_ptr: *const i64,
        dtype: i32,
        device_type: i32,
        device_index: i32,
        ret_new_tensor: &mut AtenTensorHandle,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h#L54C1-L54C145
    // AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_add_Tensor(AtenTensorHandle self, AtenTensorHandle other, double alpha, AtenTensorHandle* ret0);
    fn aoti_torch_cpu_add_Tensor(
        _self: &AtenTensorHandle,
        other: &AtenTensorHandle,
        alpha: f64,
        ret0: &mut AtenTensorHandle,
    ) -> AOTITorchError;

    // https://github.com/pytorch/pytorch/blob/274a26df346a9898b8496745c0b8b7069cb84507/torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h#L18
    fn aoti_torch_aten_fill__Scalar(_self: &AtenTensorHandle, value: f64) -> AOTITorchError;
}

#[derive(Copy, Clone, Debug)]
pub struct Device(i32);
impl Device {
    pub fn cpu() -> Self {
        Device(unsafe { aoti_torch_device_type_cpu() })
    }
    pub fn cuda() -> Self {
        Device(unsafe { aoti_torch_device_type_cuda() })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DType(i32);
impl DType {
    pub fn float32() -> Self {
        DType(unsafe { aoti_torch_dtype_float32() })
    }
}
#[derive(Copy, Clone, Debug)]
pub struct Layout(i32);
impl Layout {
    pub fn strided() -> Self {
        Layout(unsafe { aoti_torch_layout_strided() })
    }
}

pub fn main() {
    unsafe {
        println!("Hello, world!");
        println!(
            "aoti_torch_device_type_cuda: {}",
            aoti_torch_device_type_cuda()
        );
        println!(
            "aoti_torch_device_type_cpu: {}",
            aoti_torch_device_type_cpu()
        );
        // println!("aoti_torch_abi_version: {}", aoti_torch_abi_version());
        //
        // fn aoti_torch_empty_strided(
        //     ndim: i64,
        //     sizes_ptr: *const i64,
        //     strides_ptr: *const i64,
        //     dtype: i32,
        //     device_type: i32,
        //     device_index: i32,
        //     ret_new_tensor: &mut AtenTensorHandle,
        // ) -> AOTITorchError;
        let mut handle_a: AtenTensorHandle = std::ptr::null_mut();
        println!("handle_a init: {}", unsafe {
            aoti_torch_new_uninitialized_tensor(&mut handle_a)
        });
        let mut handle_b: AtenTensorHandle = std::ptr::null_mut();
        println!("handle_b init: {}", unsafe {
            aoti_torch_new_uninitialized_tensor(&mut handle_b)
        });
        let stride: i64 = 1;
        let size: i64 = 4;
        // let device = Device::cuda().0;
        let device = Device::cpu().0;
        let device_index: i32 = -1;
        let res = unsafe {
            aoti_torch_empty_strided(
                1,
                &size,
                &stride,
                DType::float32().0,
                device,
                device_index,
                &mut handle_a,
            )
        };
        println!("res: {}", res);
        let res = unsafe {
            aoti_torch_empty_strided(
                1,
                &size,
                &stride,
                DType::float32().0,
                device,
                device_index,
                &mut handle_b,
            )
        };
        println!("res: {}", res);

        let mut handle_c: AtenTensorHandle = std::ptr::null_mut();
        println!("handle_c init: {}", unsafe {
            aoti_torch_new_uninitialized_tensor(&mut handle_c)
        });
        let res = unsafe {
            aoti_torch_empty_strided(
                1,
                &size,
                &stride,
                DType::float32().0,
                device,
                device_index,
                &mut handle_c,
            )
        };
        println!("res: {}", res);
        let mut ret_device_index: i32 = 34234;
        let res = unsafe { aoti_torch_get_device_index(handle_c, &mut ret_device_index) };
        println!("handle_c res: {} ret_device_type: {ret_device_index}", res);

        // let res = unsafe { aoti_torch_cpu_add_Tensor(&handle_a, &handle_b, 1.0, &mut handle_c) };
        //

        // Maybe with scalar tensors?
        let mut handle_d: AtenTensorHandle = std::ptr::null_mut();
        println!("handle_d init: {}", unsafe {
            aoti_torch_new_uninitialized_tensor(&mut handle_d)
        });
        let res = unsafe { aoti_torch_scalar_to_tensor_float32(3.3, &mut handle_d) };
        let mut handle_e: AtenTensorHandle = std::ptr::null_mut();
        println!("handle_e init: {}", unsafe {
            aoti_torch_new_uninitialized_tensor(&mut handle_e)
        });
        let res = unsafe { aoti_torch_scalar_to_tensor_float32(5.3, &mut handle_e) };
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        println!("handle_res init: {}", unsafe {
            aoti_torch_new_uninitialized_tensor(&mut handle_res)
        });
        let res = unsafe { aoti_torch_scalar_to_tensor_float32(0.0, &mut handle_res) };
        println!("res: {}", res);

        let mut ret_device_type: i32 = 34234;
        let res = unsafe { aoti_torch_get_device_type(handle_res, &mut ret_device_type) };
        println!("res: {} ret_device_type: {ret_device_type}", res);
        let mut ret_device_index: i32 = 34234;
        let res = unsafe { aoti_torch_get_device_index(handle_res, &mut ret_device_index) };
        println!("res: {} ret_device_type: {ret_device_index}", res);

        let res =
            unsafe { aoti_torch_cpu_add_Tensor(&mut handle_d, &handle_e, 1.0, &mut handle_res) };
    }
}
