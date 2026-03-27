// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/device.h

use super::c::torch_parse_device_string;
use crate::{StableTorchResult, unsafe_call_bail};
use anyhow::{anyhow, bail};

pub use super::accelerator::DeviceIndex;
pub use crate::headeronly::core::DeviceType;
use std::ffi::CString;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Device {
    device_type: DeviceType,
    device_index: DeviceIndex,
}

impl Device {
    pub fn from_str(device_string: &str) -> StableTorchResult<Self> {
        if device_string.is_empty() {
            // This causes an assert under the hood.
            bail!("device string may not be empty");
        }
        let as_cstr = CString::new(device_string).expect("CString::new failed");
        let device_string = as_cstr.as_ptr();
        let mut device_type = 0u32;
        let mut device_index = 0i32;
        unsafe_call_bail!(torch_parse_device_string(
            device_string,
            &mut device_type,
            &mut device_index,
        ));
        Ok(Self {
            device_type: device_type.try_into()?,
            device_index: DeviceIndex(device_index),
        })
    }

    // Called 'type' on cpp.
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }
    pub fn device_index(&self) -> DeviceIndex {
        self.device_index
    }
    pub fn has_index(&self) -> bool {
        self.device_index().0 != -1
    }
    pub fn is_cuda(&self) -> bool {
        self.device_type == DeviceType::CUDA
    }
    pub fn is_cpu(&self) -> bool {
        self.device_type == DeviceType::CPU
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_device_constructor() {
        let cpu = Device::from_str("cpu").unwrap();
        assert_eq!(cpu.device_type(), DeviceType::CPU);
        let cuda_dev = Device::from_str("cuda:0").unwrap();
        assert_eq!(cuda_dev.device_type(), DeviceType::CUDA);
        let cuda_dev = Device::from_str("cuda:1").unwrap();
        assert_eq!(cuda_dev.device_type(), DeviceType::CUDA);
        assert_eq!(cuda_dev.device_index().0, 1);

        if crate::RUN_SPAMMY_TESTS {
            let res = Device::from_str("definitely_not_a_valid_type:1");
            assert!(res.is_err());
            println!("res: {res:?}");
        }
    }
}
