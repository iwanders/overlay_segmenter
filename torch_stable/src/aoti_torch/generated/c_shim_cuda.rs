use super::super::*;
// Keep the order the same as the original file.
unsafe extern "C" {

    // https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h#L53
    pub unsafe fn aoti_torch_cuda_add_Tensor(
        _self: AtenTensorHandle,
        other: AtenTensorHandle,
        alpha: f64,
        ret0: &mut AtenTensorHandle,
    ) -> AOTITorchError;
}
