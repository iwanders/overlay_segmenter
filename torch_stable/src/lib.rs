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

// The file structure follows the upstream torch repository.
// The root of this crate is https://github.com/pytorch/pytorch/tree/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc

pub mod aoti_torch;
pub mod headeronly;
pub mod stable;
mod util;

use aoti_torch::*;
// Input param: AtenTensorHandle
// Output param: &mut AtenTensorHandle

pub use util::StableTorchResult;
pub(crate) use util::unsafe_call_bail;

pub(crate) const RUN_SPAMMY_TESTS: bool = false;

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
        let mut handle_b: AtenTensorHandle = std::ptr::null_mut();
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
        let res = unsafe { aoti_torch_scalar_to_tensor_float32(3.3, &mut handle_d) };

        let mut handle_e: AtenTensorHandle = std::ptr::null_mut();
        let res = unsafe { aoti_torch_scalar_to_tensor_float32(5.3, &mut handle_e) };

        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        let res = unsafe { aoti_torch_scalar_to_tensor_float32(0.0, &mut handle_res) };
        // let res = unsafe { aoti_torch_new_uninitialized_tensor(&mut handle_res) };
        // aoti_torch_delete_tensor_object(handle_res);
        println!("res: {}", res);

        let mut ret_device_type: i32 = 34234;
        let res = unsafe { aoti_torch_get_device_type(handle_res, &mut ret_device_type) };
        println!("res: {} ret_device_type: {ret_device_type}", res);
        let mut ret_device_index: i32 = 34234;
        let res = unsafe { aoti_torch_get_device_index(handle_res, &mut ret_device_index) };
        println!("res: {} ret_device_type: {ret_device_index}", res);
        aoti_torch_delete_tensor_object(handle_res);

        let res = unsafe { aoti_torch_cpu_add_Tensor(handle_d, handle_e, 1.0, &mut handle_res) };
        let mut sum_result = 0.0;
        let res = unsafe { aoti_torch_item_float32(handle_res, &mut sum_result) };
        println!("res: {} sum_result: {sum_result}", res);

        aoti_torch_delete_tensor_object(handle_a);
        aoti_torch_delete_tensor_object(handle_b);
        aoti_torch_delete_tensor_object(handle_c);
        aoti_torch_delete_tensor_object(handle_d);
        aoti_torch_delete_tensor_object(handle_e);
        aoti_torch_delete_tensor_object(handle_res);
    }
}
