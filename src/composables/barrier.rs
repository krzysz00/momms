use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;
use composables::{Gemm3Node,AlgorithmStep};
use core::marker::PhantomData;

pub struct Barrier<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
                   S: Gemm3Node<T, At, Bt, Ct, Xt>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _xt: PhantomData<Xt>,
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
     S: Gemm3Node<T, At, Bt, Ct, Xt>>
    Gemm3Node<T, At, Bt, Ct, Xt> for Barrier<T, At, Bt, Ct, Xt, S> {
    #[inline(always)]
        unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, x: &mut Xt,
                      thr: &ThreadInfo<T>) -> () {
        thr.barrier();
        self.child.run(a, b, c, x, thr);
    }
    fn new( ) -> Barrier<T, At, Bt, Ct, Xt, S>{
        Barrier{child: S::new(), _t: PhantomData, _at: PhantomData,
                _bt: PhantomData, _ct: PhantomData, _xt: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}
