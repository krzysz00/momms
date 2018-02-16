use matrix::matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;
use std::marker::PhantomData;

// TODO: remove on non-lexical lifetimes?
// Used to prevent rather understandable borrowcheck complaints on resizing
pub struct MetadataOnlyMatrix<T: Scalar> {
    iter_height: usize,
    iter_width: usize,
    logical_w_padding: usize,
    logical_h_padding: usize,
    width: usize,
    height: usize,
    _t: PhantomData<T>,
}

impl<T: Scalar> MetadataOnlyMatrix<T> {
    pub fn new(other: &Mat<T>) -> Self{
        MetadataOnlyMatrix {
            iter_width: other.iter_width(),
            iter_height: other.iter_height(),
            logical_w_padding: other.logical_w_padding(),
            logical_h_padding: other.logical_h_padding(),
            width: other.width(),
            height: other.height(),
            _t: PhantomData,
        }
    }

    pub fn new_from_data(iter_width: usize, iter_height: usize,
                         logical_w_padding: usize, logical_h_padding: usize,
                         width: usize, height: usize) -> Self {
        MetadataOnlyMatrix {
            iter_width: iter_width,
            iter_height: iter_height,
            logical_w_padding: logical_w_padding,
            logical_h_padding: logical_h_padding,
            width: width,
            height: height,
            _t: PhantomData,
        }
    }
}

impl<T: Scalar> Mat<T> for MetadataOnlyMatrix<T> {
    #[inline(always)]
    fn get(&self, _y: usize, _x: usize) -> T { T::zero() }
    #[inline(always)]
    fn set(&mut self, _y: usize, _x: usize, _alpha: T) { }

    #[inline(always)]
    fn iter_height(&self) -> usize {
        self.iter_height
    }
    #[inline(always)]
    fn iter_width(&self) -> usize {
        self.iter_width
    }
    #[inline(always)]
    fn logical_h_padding(&self) -> usize {
        self.logical_h_padding
    }
    #[inline(always)]
    fn logical_w_padding(&self) -> usize {
        self.logical_w_padding
    }
    #[inline(always)]
    fn height(&self) -> usize {
        self.height
    }
    #[inline(always)]
    fn width(&self) -> usize {
        self.width
    }

    #[inline(always)]
    fn set_scalar(&mut self, _alpha: T) { }
    #[inline(always)]
    fn get_scalar(&self) -> T { T::zero() }
    fn push_y_split(&mut self, _start: usize, _end: usize) { }
    fn push_x_split(&mut self, _start: usize, _end: usize) { }
    #[inline(always)]
    fn pop_y_split(&mut self) { }
    #[inline(always)]
    fn pop_x_split(&mut self) { }
    fn push_y_view(&mut self, _blksz: usize) -> usize { 0 }
    fn push_x_view(&mut self, _blksz: usize) -> usize { 0 }
    #[inline(always)]
    fn pop_y_view(&mut self) { }
    #[inline(always)]
    fn pop_x_view(&mut self) { }
    fn slide_y_view_to(&mut self, _y: usize, _blksz: usize) { }
    fn slide_x_view_to(&mut self, _x: usize, _blksz: usize) {}
    #[inline(always)]
    unsafe fn make_alias(&self) -> Self { unimplemented!(); }
    #[inline(always)]
    unsafe fn send_alias(&mut self, _thr: &ThreadInfo<T>) { unimplemented!(); }
}
