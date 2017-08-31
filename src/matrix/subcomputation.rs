use thread_comm::ThreadInfo;
use matrix::{Scalar,Mat};
use super::view::{MatrixView};
use composables::{GemmNode};
use std;
use std::marker::PhantomData;

// Assumes that, when force() is called, A is m by k, B is k by n, and C is m by n
// for m and n dictated by c
// A and B might be bigger initially, and should be partitioned to the correct size
pub struct Subcomputation<'a, T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>,
                          Sc: GemmNode<T, At, Bt, Ct>>
    where Sc: Send, At: 'a, Bt: 'a, Ct: 'a {
    a: &'a mut At,
    b: &'a mut Bt,
    c: &'a mut Ct,

    subalgorithm: Sc,
    _t: PhantomData<T>,
}

impl<'a, T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, At, Bt, Ct>> Subcomputation<'a, T, At, Bt, Ct, Sc>
    where Sc: Send {

    pub fn new(a: &'a mut At, b: &'a mut Bt, c: &'a mut Ct) -> Self {
        Subcomputation {a: a, b: b, c: c,
                        subalgorithm: Sc::new(),
                        _t: PhantomData }
    }

    pub unsafe fn force(&mut self, thread_info: &ThreadInfo<T>) -> &mut Ct {
        self.subalgorithm.run(&mut self.a, &mut self.b, &mut self.c, thread_info);
        &mut self.c
    }

    pub fn get_a_scalar(&self) -> T {
        self.a.get_scalar()
    }

    pub fn set_a_scalar(&mut self, alpha: T) {
        self.a.set_scalar(alpha)
    }

    pub fn get_b_scalar(&self) -> T {
        self.b.get_scalar()
    }

    pub fn set_b_scalar(&mut self, alpha: T) {
        self.b.set_scalar(alpha)
    }

}

impl<'a, T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, At, Bt, Ct>> Mat<T> for Subcomputation<'a, T, At, Bt, Ct, Sc>
    where Sc: Send {
    // These set outputs of C
    #[inline(always)]
    fn get(&self, y: usize, x: usize) -> T {
        self.c.get(y, x)
    }

    #[inline(always)]
    fn set(&mut self, y: usize, x: usize, alpha: T) {
        self.c.set(y, x, alpha)
    }

    #[inline(always)]
    fn iter_height(&self) -> usize {
        self.b.iter_height()
    }
    #[inline(always)]
    fn iter_width(&self) -> usize {
        self.a.iter_width()
    }
    #[inline(always)]
    fn logical_h_padding(&self) -> usize {
        self.b.logical_h_padding()
    }
    #[inline(always)]
    fn logical_w_padding(&self) -> usize {
        self.a.logical_w_padding()
    }

    #[inline(always)]
    fn set_scalar(&mut self, alpha: T) {
        self.c.set_scalar(alpha)
    }

    #[inline(always)]
    fn get_scalar(&self) -> T {
        self.c.get_scalar()
    }

    fn push_y_split(&mut self, start: usize, end: usize) {
        self.a.push_y_split(start, end);
    }

    fn push_x_split(&mut self, start: usize, end: usize) {
        self.b.push_x_split(start, end);
    }

    #[inline(always)]
    fn pop_y_split(&mut self) {
        self.a.pop_y_split();
    }

    #[inline(always)]
    fn pop_x_split(&mut self) {
        self.b.pop_x_split();
    }

    fn push_y_view(&mut self, blksz: usize) -> usize{
        self.a.push_y_view(blksz)
    }

    fn push_x_view(&mut self, blksz: usize) -> usize {
        self.b.push_x_view(blksz)
    }

    #[inline(always)]
    fn pop_y_view(&mut self) {
        self.a.pop_y_view();
    }

    #[inline(always)]
    fn pop_x_view(&mut self) {
        self.b.pop_x_view();
    }

    fn slide_y_view_to(&mut self, y: usize, blksz: usize) {
        self.a.slide_y_view_to(y, blksz);
    }
    fn slide_x_view_to(&mut self, x: usize, blksz: usize) {
        self.b.slide_x_view_to(x, blksz);
    }

    #[inline(always)]
    unsafe fn make_alias(&self) -> Self {
        panic!("What do")
    }

    #[inline(always)]
    unsafe fn send_alias(&mut self, thr: &ThreadInfo<T>) {
        panic!("More what do")
    }
}
