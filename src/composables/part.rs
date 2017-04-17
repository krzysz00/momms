use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;
use composables::{Gemm3Node,AlgorithmStep};
use core::marker::PhantomData;
use typenum::Unsigned;


pub struct PartM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
                 Bsz: Unsigned, S: Gemm3Node<T, At, Bt, Ct, Xt>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _xt: PhantomData<Xt>,
    _bszt: PhantomData<Bsz>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
     Bsz: Unsigned, S: Gemm3Node<T, At, Bt, Ct, Xt>>
    Gemm3Node<T, At, Bt, Ct, Xt> for PartM<T, At, Bt, Ct, Xt, Bsz, S> {
    #[inline(always)]
        unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, x: &mut Xt,
                      thr: &ThreadInfo<T> ) -> () {
        let m_save = a.push_y_view(Bsz::to_usize());
        c.push_y_view(Bsz::to_usize());

        let mut i = 0;
        while i < m_save  {
            a.slide_y_view_to(i, Bsz::to_usize());
            c.slide_y_view_to(i, Bsz::to_usize());

            self.child.run(a, b, c, x, thr);
            i += Bsz::to_usize();
        }

        a.pop_y_view();
        c.pop_y_view();
    }
    fn new( ) -> PartM<T, At, Bt, Ct, Xt, Bsz, S>{
        PartM {child: S::new(), _t: PhantomData,
               _at: PhantomData, _bt: PhantomData,
               _ct: PhantomData, _xt: PhantomData,
               _bszt: PhantomData }
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        let mut child_desc = S::hierarchy_description();
        child_desc.push( AlgorithmStep::M{ bsz: Bsz::to_usize() } );
        child_desc
    }
}

pub struct PartN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
                 Bsz: Unsigned, S: Gemm3Node<T, At, Bt, Ct, Xt>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _xt: PhantomData<Xt>,
    _bszt: PhantomData<Bsz>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
     Bsz: Unsigned, S: Gemm3Node<T, At, Bt, Ct, Xt>>
    PartN<T, At, Bt, Ct, Xt, Bsz, S> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
     Bsz: Unsigned, S: Gemm3Node<T, At, Bt, Ct, Xt>>
    Gemm3Node<T, At, Bt, Ct, Xt> for PartN<T, At, Bt, Ct, Xt, Bsz, S> {
    #[inline(always)]
        unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, x: &mut Xt,
                        thr:&ThreadInfo<T> ) -> () {
        let n_save = b.push_x_view(Bsz::to_usize());
        c.push_x_view(Bsz::to_usize());

        let mut i = 0;
        while i < n_save {
            b.slide_x_view_to(i, Bsz::to_usize());
            c.slide_x_view_to(i, Bsz::to_usize());

            self.child.run(a, b, c, x, thr);
            i += Bsz::to_usize();
        }

        b.pop_x_view();
        c.pop_x_view();
    }
    fn new( ) -> PartN<T, At, Bt, Ct, Xt, Bsz, S>{
        PartN {child: S::new(), _t: PhantomData,
               _at: PhantomData, _bt: PhantomData,
               _ct: PhantomData, _xt: PhantomData,
               _bszt: PhantomData }
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        let mut child_desc = S::hierarchy_description();
        child_desc.push( AlgorithmStep::N{ bsz: Bsz::to_usize() } );
        child_desc
    }
}

pub struct PartK<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
                 Bsz: Unsigned, S: Gemm3Node<T, At, Bt, Ct, Xt>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _xt: PhantomData<Xt>,
    _bszt: PhantomData<Bsz>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
     Bsz: Unsigned, S: Gemm3Node<T, At, Bt, Ct, Xt>>
    PartK<T, At, Bt, Ct, Xt, Bsz, S> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
     Bsz: Unsigned, S: Gemm3Node<T, At, Bt, Ct, Xt>>
    Gemm3Node<T, At, Bt, Ct, Xt> for PartK<T, At, Bt, Ct, Xt, Bsz, S> {
    #[inline(always)]
        unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, x: &mut Xt,
                      thr: &ThreadInfo<T> ) -> () {
        let k_save = a.push_x_view(Bsz::to_usize());
        b.push_y_view(Bsz::to_usize());

        let mut i = 0;
        while i < k_save  {
            a.slide_x_view_to(i, Bsz::to_usize());
            b.slide_y_view_to(i, Bsz::to_usize());

            self.child.run(a, b, c, x, thr);
            i += Bsz::to_usize();
        }

        a.pop_x_view();
        b.pop_y_view();
    }
    fn new( ) -> PartK<T, At, Bt, Ct, Xt, Bsz, S>{
        PartK {child: S::new(), _t: PhantomData,
               _at: PhantomData, _bt: PhantomData,
               _ct: PhantomData, _xt: PhantomData,
               _bszt: PhantomData }
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        let mut child_desc = S::hierarchy_description();
        child_desc.push( AlgorithmStep::K{ bsz: Bsz::to_usize() } );
        child_desc
    }
}
