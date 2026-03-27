// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/device.h

use super::super::AOTI_TORCH_SUCCESS;
use super::accelerator::DeviceIndex;
use super::c::torch_parse_device_string;

use std::ffi::CString;
// https://github.com/pytorch/pytorch/tree/fbdef9635b009f670321b1263bec7b48e2d7379f/torch/headeronly/core

// https://github.com/pytorch/pytorch/blob/fbdef9635b009f670321b1263bec7b48e2d7379f/torch/headeronly/core/DeviceType.h#L35
// Should this be in headeronly/core/DeviceType?
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i8)]
pub enum DeviceType {
    CPU = 0,
    CUDA = 1,         // CUDA.
    MKLDNN = 2,       // Reserved for explicit MKLDNN
    OPENGL = 3,       // OpenGL
    OPENCL = 4,       // OpenCL
    IDEEP = 5,        // IDEEP.
    HIP = 6,          // AMD HIP
    FPGA = 7,         // FPGA
    MAIA = 8,         // ONNX Runtime / Microsoft
    XLA = 9,          // XLA / TPU
    Vulkan = 10,      // Vulkan
    Metal = 11,       // Metal
    XPU = 12,         // XPU
    MPS = 13,         // MPS
    Meta = 14,        // Meta (tensors with no data)
    HPU = 15,         // HPU / HABANA
    VE = 16,          // SX-Aurora / NEC
    Lazy = 17,        // Lazy Tensors
    IPU = 18,         // Graphcore IPU
    MTIA = 19,        // Meta training and inference devices
    PrivateUse1 = 20, // PrivateUse1 device
}

impl TryFrom<u32> for DeviceType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            v if v == (DeviceType::CPU as u32) => Ok(DeviceType::CPU),
            v if v == (DeviceType::CUDA as u32) => Ok(DeviceType::CUDA),
            _ => Err(()),
        }
    }
}

pub struct Device {
    device_type: DeviceType,
    device_index: DeviceIndex,
}

impl Device {
    pub fn from_str(device_string: &str) -> Result<Self, ()> {
        if device_string.is_empty() {
            // This causes an assert under the hood.
            return Err(());
        }
        let as_cstr = CString::new(device_string).expect("CString::new failed");
        let device_string = as_cstr.as_ptr();
        let mut device_type = 0u32;
        let mut device_index = 0i32;
        let res = unsafe {
            torch_parse_device_string(device_string, &mut device_type, &mut device_index)
        };
        if res == AOTI_TORCH_SUCCESS {
            Ok(Self {
                device_type: device_type.try_into()?,
                device_index: DeviceIndex(device_index),
            })
        } else {
            Err(())
        }
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
    }
}
