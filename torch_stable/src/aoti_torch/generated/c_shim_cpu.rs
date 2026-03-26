use super::super::*;
// Keep the order the same as the original file.
unsafe extern "C" {

    // https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h#L54
    // AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_add_Tensor(AtenTensorHandle self, AtenTensorHandle other, double alpha, AtenTensorHandle* ret0);
    pub unsafe fn aoti_torch_cpu_add_Tensor(
        _self: AtenTensorHandle,
        other: AtenTensorHandle,
        alpha: f64,
        ret0: &mut AtenTensorHandle,
    ) -> AOTITorchError;
}
