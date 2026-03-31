/// A Python module implemented in Rust. The name of this module must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.

// LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.13/site-packages/torch/lib python3 -m data_generator.test

// https://dmlc.github.io/dlpack/latest/python_spec.html#syntax-for-data-interchange-with-dlpack

// But it doesn't look like I can get the GPU memory pointer from tch.rs?

// If we stay on the cpu; https://github.com/PyO3/rust-numpy

#[pyo3::pymodule]
mod data_generator {
    use super::*;
    use pyo3::prelude::*;
    use tch::Tensor;

    struct TensorWrapper(Tensor);
    unsafe impl Sync for TensorWrapper {}

    #[pyclass]
    struct MyClass {
        inner: i32,
        t: TensorWrapper,
    }

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    // https://dmlc.github.io/dlpack/latest/python_spec.html#syntax-for-data-interchange-with-dlpack
    #[pyfunction]
    fn make_tensor() -> PyResult<MyClass> {
        Ok(MyClass {
            inner: 5,
            t: TensorWrapper(Tensor::from_slice(&[3, 1, 4, 1, 5])),
        })
    }
}
