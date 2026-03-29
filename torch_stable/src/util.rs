// I don't really know how to do error handling, since we only ever get a binary signal if it succeeded or not.
// THere's a check for cuda here: https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/macros.h#L9
// The c++ backtrace in the terminal seems hardcoded;
// https://github.com/pytorch/pytorch/blob/c62f72647d6f1bf86025b5e6dbdbd40e6721daf3/torch/csrc/inductor/aoti_torch/utils.h#L14

pub type StableTorchResult<T> = anyhow::Result<T>;

/// For functions that can fail somewhat gracefully.
macro_rules! unsafe_call_bail {
    ($($tokens:tt)*) => {{
        let api_call_result = unsafe {$($tokens)*};
        let code_text = stringify!($($tokens)*);
        if api_call_result == crate::AOTI_TORCH_FAILURE {
            anyhow::bail!("call failed ({}) at {}:{}", code_text, file!(), line!());
        }
    }};
}

/// For functions that cannot return a failure like Drop, or functions where it would lead to questionmark soup
/// like getting the dim() on a guaranteed existing tensor pointer.
macro_rules! unsafe_call_panic {
    ($($tokens:tt)*) => {{
        let api_call_result = unsafe {$($tokens)*};
        let code_text = stringify!($($tokens)*);
        if api_call_result == crate::AOTI_TORCH_FAILURE {
            panic!("call failed ({}) at {}:{}", code_text, file!(), line!());
        }
    }};
}

// This is a macro mostly to ensure we have the correct line number and file :/
macro_rules! unsafe_call_dispatch_bail {
    ($op_name:expr, $overload_name:expr, $stack:expr) => {{
        let op_name = std::ffi::CString::new($op_name).expect("CString::new failed");
        let op_name_cstr = op_name.as_ptr();

        let overload_name = std::ffi::CString::new($overload_name).expect("CString::new failed");
        let overload_name_cstr = overload_name.as_ptr();

        let api_call_result = unsafe {
            crate::stable::c::torch_call_dispatcher(
                op_name_cstr,
                overload_name_cstr,
                $stack.as_mut_ptr(),
                crate::TORCH_ABI_VERSION,
            )
        };
        if api_call_result == crate::AOTI_TORCH_FAILURE {
            anyhow::bail!(
                "dispatch failed ({}, {}) at {}:{}",
                $op_name,
                $overload_name,
                file!(),
                line!()
            );
        }
    }};
}

pub(crate) use unsafe_call_bail;
pub(crate) use unsafe_call_dispatch_bail;
pub(crate) use unsafe_call_panic;
