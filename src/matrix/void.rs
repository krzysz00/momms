use thread_comm::ThreadInfo;
use matrix::{Scalar,Mat};
use super::view::{MatrixView};
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
    fn off_y( &self ) -> usize {
        0
    }
    #[inline(always)]
    fn off_x( &self ) -> usize {
        0
    }

    #[inline(always)]
    fn set_off_y( &mut self, _off_y: usize ) {}
    #[inline(always)]
    fn set_off_x( &mut self, _off_x: usize ) {}
    #[inline(always)]
    fn add_off_y( &mut self, _start: usize ) {}
    #[inline(always)]
    fn add_off_x( &mut self, _start: usize ) {}

    #[inline(always)]
    fn iter_height( &self ) -> usize {
        0
    }
    #[inline(always)]
    fn iter_width( &self ) -> usize {
        0
    }
    #[inline(always)]
    fn set_iter_height( &mut self, _iter_h: usize ) {}
    #[inline(always)]
    fn set_iter_width( &mut self, _iter_w: usize ) {}

    #[inline(always)]
    fn logical_h_padding( &self ) -> usize {
        0
    }
    #[inline(always)]
    fn logical_w_padding( &self ) -> usize {
        0
    }
    #[inline(always)]
    fn set_logical_h_padding( &mut self, _h_pad: usize ) {}
    #[inline(always)]
    fn set_logical_w_padding( &mut self, _w_pad: usize ) {}

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

    fn slide_x_view_to( &mut self, _x: usize, _blksz: usize ) {}

    fn slide_y_view_to( &mut self, _y: usize, _blksz: usize ) {}

    #[inline(always)]
    unsafe fn make_alias( &self ) -> Self {
        VoidMat { _data: PhantomData }
    }

    #[inline(always)]
    unsafe fn send_alias( &mut self, _thr: &ThreadInfo<T> ) {}

}
