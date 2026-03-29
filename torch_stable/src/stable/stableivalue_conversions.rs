// https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/stableivalue_conversions.h

// This entire file is a bit freeform and doesn't follow the upstream file too precisely.
// But it just converts that C++ metaprogramming into the rust equivalent for ergonomic stack / ivalue generation.
use crate::aoti_torch::*;
use crate::stable::c::*;
use anyhow::bail;

use crate::{
    aoti_torch::AtenTensorHandle,
    headeronly::core::{DeviceType, Layout, MemoryFormat, ScalarType},
    stable::{
        device::{Device, DeviceIndex},
        scalar::Scalar,
        tensor::Tensor,
    },
    unsafe_call_panic,
};

// https://github.com/pytorch/pytorch/blob/3848e11d554a7f49925b593c40b8be0b86ac6b3f/torch/csrc/stable/stableivalue_conversions.h#L100-L101

// StableIValue is a transparent wrapper around u64, so this should work quite nicely.
pub use super::super::aoti_torch::StableIValue;

// Okay, this is quite the thing...
// https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/stableivalue_conversions.h#L224-L266
// So for nullpointers, this is a nullpointer.
// For non nullpointers a 'new StableIValue' heap allocated u64 is created, and that pointer goes into the outer
// StableIValue, we hit a snag here... we're allocating with rust, but the c++ side is clearing it up.
// Okay, so this causes mismatched free() / delete warnings in valgrind, but it may not actually be an issue?
impl<'a, T> From<&'a Option<T>> for StableIValue
where
    T: Into<StableIValue>,
    T: Copy,
{
    fn from(value: &Option<T>) -> Self {
        match value {
            Some(val) => {
                // Allocate a new u64, convert value into that.
                let converted: StableIValue = (*val).into();
                let value_u64: u64 = converted.0;
                // println!("vvalue_u64 : {value_u64:x?}");
                // if true
                // {
                //     let raw_malloc_ptr_u64 =
                //         unsafe { crate::support::iw_stable_torch_alloc_stableivalue() };
                //     unsafe { *raw_malloc_ptr_u64 = value_u64 };
                //     let ptr_as_u64: u64 = raw_malloc_ptr_u64 as u64;
                //     StableIValue(ptr_as_u64)
                // }

                {
                    let mut handle_res: *mut StableIValue = std::ptr::null_mut();
                    unsafe { aoti_torch_new_stable_ivalue(&mut handle_res) };
                    let ptr_as_u64: u64 = handle_res as u64;
                    StableIValue(ptr_as_u64)
                }

                /* else if false {
                let boxed_stable_value: Box<u64> = Box::new(value_u64);
                let stable_value = Box::into_raw(boxed_stable_value);
                let ptr_as_u64: u64 = stable_value as u64;
                StableIValue(ptr_as_u64)
                } else if false {
                use std::alloc::{Layout, System};
                let z = System;
                use std::alloc::GlobalAlloc;
                let u64_layout = Layout::new::<u64>();
                let raw_malloc_ptr = unsafe { z.alloc(u64_layout) };
                let raw_malloc_ptr_u64 = raw_malloc_ptr.cast::<u64>();
                unsafe { *raw_malloc_ptr_u64 = value_u64 };
                let ptr_as_u64: u64 = raw_malloc_ptr_u64 as u64;
                StableIValue(ptr_as_u64)
                } else {
                let raw_malloc_ptr =
                unsafe { libc::malloc(std::mem::size_of::<u64>() as libc::size_t) }
                as *mut u64;
                let raw_malloc_ptr_u64 = raw_malloc_ptr.cast::<u64>();
                unsafe { *raw_malloc_ptr_u64 = value_u64 };
                let ptr_as_u64: u64 = raw_malloc_ptr_u64 as u64;
                StableIValue(ptr_as_u64)
                }*/
            }
            None => StableIValue(0), // nullptr
        }
    }
}

// https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/stableivalue_conversions.h#L353-L378
// HeaderOnlyArrayRef is effectively a slice, so lets handle it as a slice.
impl<'a, T> From<&'a [T]> for StableIValue
// HeaderOnlyArrayRef
where
    T: Into<StableIValue>,
    T: Copy,
{
    fn from(values: &'a [T]) -> Self {
        let mut handle_res: StableListHandle = std::ptr::null_mut();
        unsafe_call_panic!(torch_new_list_reserve_size(values.len(), &mut handle_res));
        for v in values.iter() {
            unsafe_call_panic!(torch_list_push_back(handle_res, (*v).into()));
        }
        StableIValue(handle_res as u64)
    }
}

impl From<i32> for StableIValue {
    fn from(value: i32) -> Self {
        let mut res_bytes = [0u8; 8];
        let input_bytes = value.to_ne_bytes();
        res_bytes[0..input_bytes.len()].copy_from_slice(input_bytes.as_slice());
        StableIValue(u64::from_ne_bytes(res_bytes))
    }
}
impl From<bool> for StableIValue {
    fn from(value: bool) -> Self {
        let mut res_bytes = [0u8; 8];
        let input_bytes = if value { &[1] } else { &[0] };
        res_bytes[0..input_bytes.len()].copy_from_slice(input_bytes.as_slice());
        StableIValue(u64::from_ne_bytes(res_bytes))
    }
}

// THis is all wrong :(
// https://github.com/pytorch/pytorch/blob/3848e11d554a7f49925b593c40b8be0b86ac6b3f/torch/csrc/stable/stableivalue_conversions.h#L83
// In combination with
// https://github.com/pytorch/pytorch/blob/3848e11d554a7f49925b593c40b8be0b86ac6b3f/torch/csrc/stable/stableivalue_conversions.h#L113-L114
// THey're all 32 bit integers, and copied into the leading values.

impl From<&Tensor> for StableIValue {
    fn from(value: &Tensor) -> Self {
        // https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/notes/libtorch_stable_abi.md?plain=1#L167
        // Is this even right?
        // | torch::stable::Tensor | raw bitwise copy of underlying AtenTensorHandle into leading bytes of uint64_t | at::Tensor |  Tensor |
        // conflicts with:
        // https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/stableivalue_conversions.h#L277
        // which does;
        // AtenTensorHandle new_ath;
        // TORCH_ERROR_CODE_CHECK(aoti_torch_new_tensor_handle(val.get(), &new_ath));
        // return torch::stable::detail::from(new_ath);

        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_panic!(aoti_torch_new_tensor_handle(value.get(), &mut handle_res));
        StableIValue(handle_res as u64)
    }
}

impl From<ScalarType> for StableIValue {
    fn from(value: ScalarType) -> Self {
        Self(value as u64)
    }
}

impl From<DeviceType> for StableIValue {
    fn from(value: DeviceType) -> Self {
        value.to_constant().into()
    }
}

impl From<MemoryFormat> for StableIValue {
    fn from(value: MemoryFormat) -> Self {
        Self(value as u64)
    }
}
impl From<Layout> for StableIValue {
    fn from(value: Layout) -> Self {
        value.to_constant().into()
    }
}

impl From<i64> for StableIValue {
    fn from(value: i64) -> Self {
        let bitwise_value: u64 = u64::from_ne_bytes(value.to_ne_bytes());
        Self(bitwise_value)
    }
}
impl From<usize> for StableIValue {
    fn from(value: usize) -> Self {
        let bitwise_value: u64 = u64::from_ne_bytes(value.to_ne_bytes());
        Self(bitwise_value)
    }
}

impl From<f64> for StableIValue {
    fn from(value: f64) -> Self {
        Self(value.to_bits())
    }
}
impl From<DeviceIndex> for StableIValue {
    fn from(value: DeviceIndex) -> Self {
        let bitwise_value: u64 = u32::from_ne_bytes((value.0 as i32).to_ne_bytes()) as u64;
        Self(bitwise_value)
    }
}

impl From<Scalar> for StableIValue {
    fn from(value: Scalar) -> Self {
        let mut handle_res: StableListHandle = std::ptr::null_mut();
        unsafe_call_panic!(torch_new_list_reserve_size(2, &mut handle_res));

        let scalar_type = value.scalar_type();
        let scalar_value = value.value();

        unsafe_call_panic!(torch_list_push_back(handle_res, scalar_type.into()));
        unsafe_call_panic!(torch_list_push_back(handle_res, scalar_value.into()));

        Self(handle_res as u64)
    }
}

// https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/stable/stableivalue_conversions.h#L400-L414
impl From<Device> for StableIValue {
    fn from(value: Device) -> Self {
        // Pack: lower 32 bits = device index, upper 32 bits = device type (shim)
        let device_index_shim: StableIValue = value.device_index().into();
        let device_index_bits: u64 = device_index_shim.0;
        // println!("value.device_type(): {:?}", value.device_type());
        let device_type_shim: StableIValue = value.device_type().into();
        // println!("value.device_type_shim(): {:?}", device_type_shim);
        let device_type_bits: u64 = device_type_shim.0 << 32;
        // Device is made up of left: ffffffff | 0
        // println!("device_index_bits: {device_index_bits:x?} | {device_type_bits:x?}");
        Self(device_index_bits | device_type_bits)
    }
}

impl TryFrom<StableIValue> for Tensor {
    type Error = anyhow::Error;

    fn try_from(value: StableIValue) -> Result<Self, Self::Error> {
        if value.0 == 0 {
            bail!("failed to convert StableIValue to Tensor; nullptr");
        }
        /*
        let handle: *mut AtenTensorHandle = value.0 as *mut AtenTensorHandle;
        if handle.is_null() {
            bail!("failed to convert StableIValue nullptr to Tensor");
        }
        Ok(Tensor::from_handle(unsafe { *handle }))
        */
        let handle: AtenTensorHandle = value.0 as AtenTensorHandle;
        if handle.is_null() {
            bail!("failed to convert StableIValue nullptr to Tensor");
        }
        Ok(Tensor::from_handle(unsafe { handle }))
    }
}
