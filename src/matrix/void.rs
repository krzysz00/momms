use thread_comm::ThreadInfo;
use matrix::{Scalar,Mat};
use core::marker::PhantomData;

pub struct VoidMat<T: Scalar> {
    _data: PhantomData<T>
}

impl<T: Scalar> VoidMat<T> {
    pub fn new() -> Self {
        VoidMat { _data: PhantomData }
    }
}

impl<T: Scalar> Mat<T> for VoidMat<T> {
    #[inline(always)]
    fn get( &self, _y: usize, _x: usize) -> T {
        panic!("Can't get on void matrix")
    }
    #[inline(always)]
    fn set( &mut self, _y: usize, _x: usize, _alpha: T)
        where T: Sized {
        panic!("Can't set on void matrix")
    }

    #[inline(always)]
    fn iter_height( &self ) -> usize {
        0
    }
    #[inline(always)]
    fn iter_width( &self ) -> usize {
        0
    }

    #[inline(always)]
    fn logical_h_padding( &self ) -> usize {
        0
    }
    #[inline(always)]
    fn logical_w_padding( &self ) -> usize {
        0
    }

    #[inline(always)]
    fn set_scalar(&mut self, _alpha: T) {}

    #[inline(always)]
    fn get_scalar(&self) -> T {
        T::zero()
    }

    fn push_x_view( &mut self, _blksz: usize ) -> usize {
        0
    }

    fn push_y_view( &mut self, _blksz: usize ) -> usize{
        0
    }
    #[inline(always)]
    fn pop_x_view( &mut self ) {}
    #[inline(always)]
    fn pop_y_view( &mut self ) {}

    fn slide_x_view_to(&mut self, _x: usize, _blksz: usize) {}

    fn slide_y_view_to(&mut self, _y: usize, _blksz: usize) {}

    fn push_y_split(&mut self, _start: usize, _end: usize) {}
    fn push_x_split(&mut self, _start: usize, _end: usize) {}
    fn pop_y_split(&mut self) {}
    fn pop_x_split(&mut self) {}

    #[inline(always)]
    unsafe fn make_alias( &self ) -> Self {
        VoidMat { _data: PhantomData }
    }

    #[inline(always)]
    unsafe fn send_alias( &mut self, _thr: &ThreadInfo<T> ) {}

}
