use thread_comm::ThreadInfo;
use matrix::{Scalar,Mat};
use std::marker::PhantomData;
use std::convert::AsMut;

// Assumes that, when force() is called, A is m by k, B is k by n, and C is m by n
// for m and n dictated by c
// A and B might be bigger initially, and should be partitioned to the correct size
pub struct Subcomputation<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> {
    pub a: At,
    pub b: Bt,
    pub c: Ct,

    _t: PhantomData<T>,
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> Subcomputation<T, At, Bt, Ct> {
    pub fn new(a: At, b: Bt, c: Ct) -> Self {
        Subcomputation {a: a, b: b, c: c,
                        _t: PhantomData }
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

    // drops old B
    pub fn set_b<NewBt: Mat<T>>(self, new_b: NewBt) -> Subcomputation<T, At, NewBt, Ct> {
        let old_b = self.b;
        let ret = Subcomputation {a: self.a, b: new_b, c: self.c,
                                  _t: PhantomData};
        ::std::mem::drop(old_b);
        ret
    }

    // drops old C
    pub fn set_c<NewCt: Mat<T>>(self, new_c: NewCt) -> Subcomputation<T, At, Bt, NewCt> {
        let old_c = self.c;
        let ret = Subcomputation {a: self.a, b: self.b, c: new_c,
                                  _t: PhantomData};
        ::std::mem::drop(old_c);
        ret
    }

    pub fn inner_transpose(self) -> TransposingSubcomputation<T, Bt, At, Ct> {
        TransposingSubcomputation::new(self.b, self.a, self.c)
    }
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>>
    Mat<T> for Subcomputation<T, At, Bt, Ct>
{
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
        self.a.iter_height()
    }
    #[inline(always)]
    fn iter_width(&self) -> usize {
        self.b.iter_width()
    }
    #[inline(always)]
    fn logical_h_padding(&self) -> usize {
        self.a.logical_h_padding()
    }
    #[inline(always)]
    fn logical_w_padding(&self) -> usize {
        self.b.logical_w_padding()
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
    unsafe fn send_alias(&mut self, _thr: &ThreadInfo<T>) {
        panic!("More what do")
    }
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>>
    AsMut<Subcomputation<T, At, Bt, Ct>>
    for Subcomputation<T, At, Bt, Ct> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut Subcomputation<T, At, Bt, Ct> {
        self
    }
}

// Conceptually, holds an m x n matrix (which may be computed in parts)
// as far as the outside world is concerned
// However, the internal algorithm is of the form (n x k * k x m)^T
// So partitions in the m dimension go to the second matrix, not the first one
pub struct TransposingSubcomputation<T: Scalar, Btt: Mat<T>, Att: Mat<T>, Ct: Mat<T>>
    (pub Subcomputation<T, Btt, Att, Ct>);

impl<T: Scalar, Btt: Mat<T>, Att: Mat<T>, Ct: Mat<T>> TransposingSubcomputation<T, Btt, Att, Ct> {
    pub fn new(bt: Btt, at: Att, c: Ct) -> Self {
        TransposingSubcomputation(Subcomputation {a: bt, b: at, c: c,
                        _t: PhantomData })
    }

    pub fn inner_transpose(self) -> Subcomputation<T, Att, Btt, Ct> {
        Subcomputation::new(self.0.b, self.0.a, self.0.c)
    }
}

impl<T: Scalar, Btt: Mat<T>, Att: Mat<T>, Ct: Mat<T>>
    Mat<T> for TransposingSubcomputation<T, Btt, Att, Ct>
{
    // These set outputs of C
    #[inline(always)]
    fn get(&self, y: usize, x: usize) -> T {
        self.0.c.get(y, x)
    }

    #[inline(always)]
    fn set(&mut self, y: usize, x: usize, alpha: T) {
        self.0.c.set(y, x, alpha)
    }

    #[inline(always)]
    fn iter_height(&self) -> usize {
        self.0.b.iter_width()
    }
    #[inline(always)]
    fn iter_width(&self) -> usize {
        self.0.a.iter_height()
    }
    #[inline(always)]
    fn logical_h_padding(&self) -> usize {
        self.0.b.logical_w_padding()
    }
    #[inline(always)]
    fn logical_w_padding(&self) -> usize {
        self.0.a.logical_h_padding()
    }

    #[inline(always)]
    fn set_scalar(&mut self, alpha: T) {
        self.0.c.set_scalar(alpha)
    }

    #[inline(always)]
    fn get_scalar(&self) -> T {
        self.0.c.get_scalar()
    }

    fn push_y_split(&mut self, start: usize, end: usize) {
        self.0.b.push_x_split(start, end);
    }

    fn push_x_split(&mut self, start: usize, end: usize) {
        self.0.a.push_y_split(start, end);
    }

    #[inline(always)]
    fn pop_y_split(&mut self) {
        self.0.b.pop_x_split();
    }

    #[inline(always)]
    fn pop_x_split(&mut self) {
        self.0.a.pop_y_split();
    }

    fn push_y_view(&mut self, blksz: usize) -> usize{
        self.0.b.push_x_view(blksz)
    }

    fn push_x_view(&mut self, blksz: usize) -> usize {
        self.0.a.push_y_view(blksz)
    }

    #[inline(always)]
    fn pop_y_view(&mut self) {
        self.0.b.pop_x_view();
    }

    #[inline(always)]
    fn pop_x_view(&mut self) {
        self.0.a.pop_y_view();
    }

    fn slide_y_view_to(&mut self, y: usize, blksz: usize) {
        self.0.b.slide_x_view_to(y, blksz);
    }
    fn slide_x_view_to(&mut self, x: usize, blksz: usize) {
        self.0.a.slide_y_view_to(x, blksz);
    }

    #[inline(always)]
    unsafe fn make_alias(&self) -> Self {
        panic!("What do")
    }

    #[inline(always)]
    unsafe fn send_alias(&mut self, _thr: &ThreadInfo<T>) {
        panic!("More what do")
    }
}

impl<T: Scalar, Btt: Mat<T>, Att: Mat<T>, Ct: Mat<T>>
    AsMut<Subcomputation<T, Btt, Att, Ct>>
    for TransposingSubcomputation<T, Btt, Att, Ct> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut Subcomputation<T, Btt, Att, Ct> {
        &mut self.0
    }
}
