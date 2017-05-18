use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;

#[derive(Copy, Clone)]
pub enum AlgorithmStep {
    M { bsz: usize },
    N { bsz: usize },
    K { bsz: usize },
}

pub trait Gemm3Node<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, x: &mut Xt,
                    thr: &ThreadInfo<T> ) -> ();
    fn new() -> Self;
    fn hierarchy_description() -> Vec<AlgorithmStep>;
}
