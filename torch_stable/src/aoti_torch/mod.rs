mod c;
mod generated;
// C api's don't have namespaces, to be robust against moving stuff between files, we export everything into this
// bin.
pub use c::*;
pub use generated::*;
