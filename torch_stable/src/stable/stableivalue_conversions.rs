// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/stableivalue_conversions.h

use anyhow::bail;

use crate::{
    aoti_torch::AtenTensorHandle,
    headeronly::core::{DeviceType, Layout, MemoryFormat, ScalarType},
    stable::{
        device::{Device, DeviceIndex},
        tensor::Tensor,
    },
};

// https://github.com/pytorch/pytorch/blob/3848e11d554a7f49925b593c40b8be0b86ac6b3f/torch/csrc/stable/stableivalue_conversions.h#L100-L101

// This is a bit freeform...
// But it just converts that C++ metaprogramming into the rust equivalent for ergonomic stack / ivalue generation.
// StableIValue is a transparent wrapper around u64, so this should work quite nicely.
pub use super::super::aoti_torch::StableIValue;

impl<'a, T> From<&'a Option<T>> for StableIValue
where
    T: Into<StableIValue>,
{
    fn from(value: &Option<T>) -> Self {
        match value {
            Some(val) => {
                //  Ref to pointer, then to u64.
                let ptr: *const T = val as *const T;
                let ptr_as_u64: u64 = ptr as u64;
                StableIValue(ptr_as_u64)
            }
            None => StableIValue(0), // nullptr
        }
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
        Self(value.get() as u64)
    }
}

impl From<ScalarType> for StableIValue {
    fn from(value: ScalarType) -> Self {
        Self(value as u64)
    }
}

impl From<DeviceType> for StableIValue {
    fn from(value: DeviceType) -> Self {
        Self(value as u64)
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
        let bitwise_value: u64 = u64::from_ne_bytes((value.0 as i64).to_ne_bytes());
        Self(bitwise_value)
    }
}

// https://github.com/pytorch/pytorch/blob/3848e11d554a7f49925b593c40b8be0b86ac6b3f/torch/csrc/stable/stableivalue_conversions.h#L400-L414
impl From<Device> for StableIValue {
    fn from(value: Device) -> Self {
        // Pack: lower 32 bits = device index, upper 32 bits = device type (shim)
        let device_type_shim: StableIValue = value.device_type().into();
        let device_index_shim: StableIValue = value.device_index().into();
        let device_index_bits: u64 = device_index_shim.0;
        let device_type_bits: u64 = device_type_shim.0 << 32;
        Self(device_index_bits | device_type_bits)
    }
}

impl TryFrom<StableIValue> for Tensor {
    type Error = anyhow::Error;

    fn try_from(value: StableIValue) -> Result<Self, Self::Error> {
        if value.0 == 0 {
            bail!("failed to convert StableIValue to Tensor; nullptr");
        }
        let handle: AtenTensorHandle = value.0 as AtenTensorHandle;
        Ok(Tensor::from_handle(handle))
    }
}
