use anyhow::anyhow;

use crate::aoti_torch::*;

// https://github.com/pytorch/pytorch/blob/3848e11d554a7f49925b593c40b8be0b86ac6b3f/torch/csrc/stable/stableivalue_conversions.h#L100-L101
// Gaaah, these are the header side values only, they always call through the shim to get the actual values.
// TODO:
// Something like:
// static DEVICE_TYPE_CPU: i8  = unsafe{aoti_torch_device_type_cpu()};
// Eliminate the constants from this file completely.

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
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            v if v == (DeviceType::CPU as u32) => Ok(DeviceType::CPU),
            v if v == (DeviceType::CUDA as u32) => Ok(DeviceType::CUDA),
            v if v == (DeviceType::MKLDNN as u32) => Ok(DeviceType::MKLDNN),
            v if v == (DeviceType::OPENGL as u32) => Ok(DeviceType::OPENGL),
            v if v == (DeviceType::OPENCL as u32) => Ok(DeviceType::OPENCL),
            v if v == (DeviceType::IDEEP as u32) => Ok(DeviceType::IDEEP),
            v if v == (DeviceType::HIP as u32) => Ok(DeviceType::HIP),
            v if v == (DeviceType::FPGA as u32) => Ok(DeviceType::FPGA),
            v if v == (DeviceType::MAIA as u32) => Ok(DeviceType::MAIA),
            v if v == (DeviceType::XLA as u32) => Ok(DeviceType::XLA),
            v if v == (DeviceType::Vulkan as u32) => Ok(DeviceType::Vulkan),
            v if v == (DeviceType::Metal as u32) => Ok(DeviceType::Metal),
            v if v == (DeviceType::XPU as u32) => Ok(DeviceType::XPU),
            v if v == (DeviceType::MPS as u32) => Ok(DeviceType::MPS),
            v if v == (DeviceType::Meta as u32) => Ok(DeviceType::Meta),
            v if v == (DeviceType::HPU as u32) => Ok(DeviceType::HPU),
            v if v == (DeviceType::VE as u32) => Ok(DeviceType::VE),
            v if v == (DeviceType::Lazy as u32) => Ok(DeviceType::Lazy),
            v if v == (DeviceType::IPU as u32) => Ok(DeviceType::IPU),
            v if v == (DeviceType::MTIA as u32) => Ok(DeviceType::MTIA),
            v if v == (DeviceType::PrivateUse1 as u32) => Ok(DeviceType::PrivateUse1),
            _ => Err(anyhow!("could not convert {} into DeviceType", value)),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i8)]
pub enum Layout {
    Strided,
    Sparse,
    SparseCsr,
    Mkldnn,
    SparseCsc,
    SparseBsr,
    SparseBsc,
    Jagged,
    // NumOptions,
}
impl Layout {
    pub fn to_constant(&self) -> i32 {
        unsafe {
            match self {
                Layout::Strided => aoti_torch_layout_strided(),
                Layout::Sparse => aoti_torch_layout_sparse_coo(),
                Layout::SparseCsr => aoti_torch_layout_sparse_csr(),
                Layout::SparseCsc => aoti_torch_layout_sparse_csc(),
                Layout::SparseBsr => aoti_torch_layout_sparse_bsr(),
                Layout::SparseBsc => aoti_torch_layout_sparse_bsc(),
                Layout::Mkldnn => aoti_torch_layout__mkldnn(),
                Layout::Jagged => aoti_torch_layout_jagged(),
            }
        }
    }
}

impl TryFrom<i32> for Layout {
    type Error = anyhow::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            v if v == (Layout::Strided as i32) => Ok(Layout::Strided),
            v if v == (Layout::Sparse as i32) => Ok(Layout::Sparse),
            v if v == (Layout::SparseCsr as i32) => Ok(Layout::SparseCsr),
            v if v == (Layout::Mkldnn as i32) => Ok(Layout::Mkldnn),
            v if v == (Layout::SparseCsc as i32) => Ok(Layout::SparseCsc),
            v if v == (Layout::SparseBsr as i32) => Ok(Layout::SparseBsr),
            v if v == (Layout::SparseBsc as i32) => Ok(Layout::SparseBsc),
            v if v == (Layout::Jagged as i32) => Ok(Layout::Jagged),
            _ => Err(anyhow!("could not convert {} into Layout", value)),
        }
    }
}
// ScalarType
// Tostring; https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/headeronly/core/ScalarType.h#L320
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/headeronly/core/ScalarType.h#L258-L264
// List is here: https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/headeronly/core/ScalarType.h#L103
//
// Lets go with the safe solution:
// /tmp/pytorch$ touch torch/headeronly/macros/cmake_macros.h
// /tmp/pytorch$ cat test.cpp
// #include "torch/headeronly/core/ScalarType.h"
// int main(){
// }
// /tmp/pytorch$ gcc -I. -E test.cpp -o test.o
// And then search for 'enum class ScalarType' in that test.o file.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i8)]
#[allow(non_camel_case_types)]
pub enum ScalarType {
    Byte,
    Char,
    Short,
    Int,
    Long,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,
    QUInt4x2,
    QUInt2x4,
    Bits1x8,
    Bits2x4,
    Bits4x2,
    Bits8,
    Bits16,
    Float8_e5m2,
    Float8_e4m3fn,
    Float8_e5m2fnuz,
    Float8_e4m3fnuz,
    UInt16,
    UInt32,
    UInt64,
    UInt1,
    UInt2,
    UInt3,
    UInt4,
    UInt5,
    UInt6,
    UInt7,
    Int1,
    Int2,
    Int3,
    Int4,
    Int5,
    Int6,
    Int7,
    Float8_e8m0fnu,
    Float4_e2m1fn_x2,
    Undefined,
    // NumOptions,
}

impl TryFrom<i32> for ScalarType {
    type Error = anyhow::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            v if v == (ScalarType::Byte as i32) => Ok(ScalarType::Byte),
            v if v == (ScalarType::Char as i32) => Ok(ScalarType::Char),
            v if v == (ScalarType::Short as i32) => Ok(ScalarType::Short),
            v if v == (ScalarType::Byte as i32) => Ok(ScalarType::Byte),
            v if v == (ScalarType::Char as i32) => Ok(ScalarType::Char),
            v if v == (ScalarType::Short as i32) => Ok(ScalarType::Short),
            v if v == (ScalarType::Int as i32) => Ok(ScalarType::Int),
            v if v == (ScalarType::Long as i32) => Ok(ScalarType::Long),
            v if v == (ScalarType::Half as i32) => Ok(ScalarType::Half),
            v if v == (ScalarType::Float as i32) => Ok(ScalarType::Float),
            v if v == (ScalarType::Double as i32) => Ok(ScalarType::Double),
            v if v == (ScalarType::ComplexHalf as i32) => Ok(ScalarType::ComplexHalf),
            v if v == (ScalarType::ComplexFloat as i32) => Ok(ScalarType::ComplexFloat),
            v if v == (ScalarType::ComplexDouble as i32) => Ok(ScalarType::ComplexDouble),
            v if v == (ScalarType::Bool as i32) => Ok(ScalarType::Bool),
            v if v == (ScalarType::QInt8 as i32) => Ok(ScalarType::QInt8),
            v if v == (ScalarType::QUInt8 as i32) => Ok(ScalarType::QUInt8),
            v if v == (ScalarType::QInt32 as i32) => Ok(ScalarType::QInt32),
            v if v == (ScalarType::BFloat16 as i32) => Ok(ScalarType::BFloat16),
            v if v == (ScalarType::QUInt4x2 as i32) => Ok(ScalarType::QUInt4x2),
            v if v == (ScalarType::QUInt2x4 as i32) => Ok(ScalarType::QUInt2x4),
            v if v == (ScalarType::Bits1x8 as i32) => Ok(ScalarType::Bits1x8),
            v if v == (ScalarType::Bits2x4 as i32) => Ok(ScalarType::Bits2x4),
            v if v == (ScalarType::Bits4x2 as i32) => Ok(ScalarType::Bits4x2),
            v if v == (ScalarType::Bits8 as i32) => Ok(ScalarType::Bits8),
            v if v == (ScalarType::Bits16 as i32) => Ok(ScalarType::Bits16),
            v if v == (ScalarType::Float8_e5m2 as i32) => Ok(ScalarType::Float8_e5m2),
            v if v == (ScalarType::Float8_e4m3fn as i32) => Ok(ScalarType::Float8_e4m3fn),
            v if v == (ScalarType::Float8_e5m2fnuz as i32) => Ok(ScalarType::Float8_e5m2fnuz),
            v if v == (ScalarType::Float8_e4m3fnuz as i32) => Ok(ScalarType::Float8_e4m3fnuz),
            v if v == (ScalarType::UInt16 as i32) => Ok(ScalarType::UInt16),
            v if v == (ScalarType::UInt32 as i32) => Ok(ScalarType::UInt32),
            v if v == (ScalarType::UInt64 as i32) => Ok(ScalarType::UInt64),
            v if v == (ScalarType::UInt1 as i32) => Ok(ScalarType::UInt1),
            v if v == (ScalarType::UInt2 as i32) => Ok(ScalarType::UInt2),
            v if v == (ScalarType::UInt3 as i32) => Ok(ScalarType::UInt3),
            v if v == (ScalarType::UInt4 as i32) => Ok(ScalarType::UInt4),
            v if v == (ScalarType::UInt5 as i32) => Ok(ScalarType::UInt5),
            v if v == (ScalarType::UInt6 as i32) => Ok(ScalarType::UInt6),
            v if v == (ScalarType::UInt7 as i32) => Ok(ScalarType::UInt7),
            v if v == (ScalarType::Int1 as i32) => Ok(ScalarType::Int1),
            v if v == (ScalarType::Int2 as i32) => Ok(ScalarType::Int2),
            v if v == (ScalarType::Int3 as i32) => Ok(ScalarType::Int3),
            v if v == (ScalarType::Int4 as i32) => Ok(ScalarType::Int4),
            v if v == (ScalarType::Int5 as i32) => Ok(ScalarType::Int5),
            v if v == (ScalarType::Int6 as i32) => Ok(ScalarType::Int6),
            v if v == (ScalarType::Int7 as i32) => Ok(ScalarType::Int7),
            v if v == (ScalarType::Float8_e8m0fnu as i32) => Ok(ScalarType::Float8_e8m0fnu),
            v if v == (ScalarType::Float4_e2m1fn_x2 as i32) => Ok(ScalarType::Float4_e2m1fn_x2),
            v if v == (ScalarType::Undefined as i32) => Ok(ScalarType::Undefined),
            _ => Err(anyhow!("could not convert {} into ScalarType", value)),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i8)]
#[allow(non_camel_case_types)]
pub enum MemoryFormat {
    Contiguous,
    Preserve,
    ChannelsLast,
    ChannelsLast3d,
    // NumOptions,
}

impl TryFrom<i32> for MemoryFormat {
    type Error = anyhow::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            v if v == (MemoryFormat::Contiguous as i32) => Ok(MemoryFormat::Contiguous),
            v if v == (MemoryFormat::Preserve as i32) => Ok(MemoryFormat::Preserve),
            v if v == (MemoryFormat::ChannelsLast as i32) => Ok(MemoryFormat::ChannelsLast),
            v if v == (MemoryFormat::ChannelsLast3d as i32) => Ok(MemoryFormat::ChannelsLast3d),
            _ => Err(anyhow!("could not convert {} into ChannelsLast3d", value)),
        }
    }
}
