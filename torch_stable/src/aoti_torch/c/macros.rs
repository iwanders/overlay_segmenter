// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/c/macros.h#L32
#[repr(C)]
pub struct AtenTensorOpaque {
    _private: [u8; 0],
}
pub type AtenTensorHandle = *mut AtenTensorOpaque;
pub type AOTITorchError = i32;

pub const AOTI_TORCH_SUCCESS: i32 = 0;
pub const AOTI_TORCH_FAILURE: i32 = 1;
