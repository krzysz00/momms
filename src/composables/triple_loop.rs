pub use matrix::{Scalar,Mat};
pub use composables::{Gemm3Node,AlgorithmStep};
pub use thread_comm::ThreadInfo;

pub struct TripleLoop{}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>>
    Gemm3Node<T, At, Bt, Ct, Xt> for TripleLoop {
    #[inline(always)]
        unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct,
                      _x: &mut Xt, _thr: &ThreadInfo<T> ) -> () {
        //For now, let's do an axpy based gemm
        for x in 0..c.width() {
            for z in 0..a.width() {
                for y in 0..c.height() {
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set(y, x, t);
                }
            }
        }
    }
    fn new() -> Self { TripleLoop{} }
    fn hierarchy_description() -> Vec<AlgorithmStep> { Vec::new() }  
}
