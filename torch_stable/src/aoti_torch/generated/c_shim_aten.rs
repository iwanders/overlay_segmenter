use super::super::*;
// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h
unsafe extern "C" {
    // https://github.com/pytorch/pytorch/blob/274a26df346a9898b8496745c0b8b7069cb84507/torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h#L18
    pub unsafe fn aoti_torch_aten_fill__Scalar(
        _self: AtenTensorHandle,
        value: f64,
    ) -> AOTITorchError;
}
