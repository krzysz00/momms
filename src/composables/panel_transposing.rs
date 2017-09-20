use matrix::{Scalar,Mat,RoCM,ResizableBuffer,PanelTranspose};
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};
use std::marker::PhantomData;

pub struct TransposingOutputPanel<T: Scalar, At: Mat<T>, Bt: Mat<T>,
                                  Ct: Mat<T> + ResizableBuffer<T> + RoCM<T> + PanelTranspose<T, Ct, SubT>,
                                  SubT: Mat<T> + ResizableBuffer<T> + RoCM<T> + PanelTranspose<T, SubT, Ct>,
                                  S: GemmNode<T, At, Bt, SubT>> {
    child: S,

    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _subt: PhantomData<SubT>,
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>,
     Ct: Mat<T> + ResizableBuffer<T> + RoCM<T> + PanelTranspose<T, Ct, SubT>,
     SubT: Mat<T> + ResizableBuffer<T> + RoCM<T> + PanelTranspose<T, SubT, Ct>,
     S: GemmNode<T, At, Bt, SubT>> GemmNode<T, At, Bt, Ct>
    for TransposingOutputPanel<T, At, Bt, Ct, SubT, S>
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct,
                  thr: &ThreadInfo<T>)  {
        let mut new_c = c.transposing_clone();
        self.child.run(a, b, &mut new_c, thr);
        c.reintegrate(new_c)
    }

    fn new() -> Self {
        TransposingOutputPanel { child: S::new(),
                                 _t: PhantomData, _at: PhantomData,
                                 _bt: PhantomData, _ct: PhantomData, _subt: PhantomData }
    }

    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}
