// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/stableivalue_conversions.h

use crate::{
    headeronly::core::{DeviceType, Layout, MemoryFormat, ScalarType},
    stable::{device::Device, tensor::Tensor},
};

// https://github.com/pytorch/pytorch/blob/3848e11d554a7f49925b593c40b8be0b86ac6b3f/torch/csrc/stable/stableivalue_conversions.h#L100-L101

// This is a bit freeform...
// But it just converts that C++ metaprogramming into the rust equivalent for ergonomic stack / ivalue generation.
// StableIValue is a transparent wrapper around u64, so this should work quite nicely.
use super::super::aoti_torch::StableIValue;

impl<'a, T> From<Option<&'a T>> for StableIValue
where
    &'a T: Into<StableIValue>,
{
    fn from(value: Option<&'a T>) -> Self {
        match value {
            Some(val) => {
                //  Ref to pointer, then to u64.
                let ptr: *const T = val;
                let ptr_as_u64: u64 = ptr as u64;
                StableIValue(ptr_as_u64)
            }
            None => StableIValue(0), // nullptr
        }
    }
}

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
        Self(value as u64)
    }
}

impl From<bool> for StableIValue {
    fn from(value: bool) -> Self {
        Self(value as u64)
    }
}

impl From<i64> for StableIValue {
    fn from(value: i64) -> Self {
        let bitwise_value: u64 = u64::from_ne_bytes(value.to_ne_bytes());
        Self(bitwise_value)
    }
}

impl From<f64> for StableIValue {
    fn from(value: f64) -> Self {
        Self(value.to_bits())
    }
}

impl From<Device> for StableIValue {
    fn from(value: Device) -> Self {
        Self(value.to_bits())
    }
}
