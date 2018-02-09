#[cfg(feature="hsw")]
mod hsw;

#[cfg(feature="hsw")]
pub use self::hsw::{
    GotoDgemm, GotoDgemm3Sub, GotoDgemm3,
    ColPM, RowPM,
    Dgemm3Sub, Dgemm3TmpWidth, Dgemm3TmpHeight,
    Dgemm3,
};
