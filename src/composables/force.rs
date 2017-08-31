use matrix::{Scalar,Mat,Subcomputation};
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};
use std::marker::PhantomData;

pub struct ForceA<T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
                  Bt: Mat<T>, Ct: Mat<T>,
                  Sc: GemmNode<T, AiT, BiT, CiT>,
                  S: GemmNode<T, CiT, Bt, Ct>> {
    child: S,
    subalgorithm: Sc,
    _t: PhantomData<T>,
    _ait: PhantomData<AiT>,
    _bit: PhantomData<BiT>,
    _cit: PhantomData<CiT>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}

impl<T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>, Bt: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>, S: GemmNode<T, CiT, Bt, Ct>>
    ForceA<T, AiT, BiT, CiT, Bt, Ct, Sc, S> {
}

impl<T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     Bt: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>,
     S: GemmNode<T, CiT, Bt, Ct>>
    GemmNode<T, Subcomputation<T, AiT, BiT, CiT>, Bt, Ct>
    for ForceA<T, AiT, BiT, CiT, Bt, Ct, Sc, S> where Sc: Send
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut Subcomputation<T, AiT, BiT, CiT>,
                  b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        self.subalgorithm.run(&mut a.a, &mut a.b, &mut a.c, thr);
        let new_a = &mut a.c;
        self.child.run(new_a, b, c, thr);
    }

    fn new() -> Self {
        ForceA {child: S::new(), subalgorithm: Sc::new(),
                _t: PhantomData, _ait: PhantomData,
                _bit: PhantomData, _cit: PhantomData,
                _bt: PhantomData, _ct: PhantomData }
    }

    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}

pub struct ForceB<T: Scalar, At: Mat<T>,
                  AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
                  Ct: Mat<T>,
                  Sc: GemmNode<T, AiT, BiT, CiT>,
                  S: GemmNode<T, At, CiT, Ct>> {
    child: S,
    subalgorithm: Sc,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _ait: PhantomData<AiT>,
    _bit: PhantomData<BiT>,
    _cit: PhantomData<CiT>,
    _ct: PhantomData<Ct>,
}

impl<T: Scalar, At: Mat<T>,
     AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>, S: GemmNode<T, At, CiT, Ct>>
    ForceB<T, At, AiT, BiT, CiT, Ct, Sc, S> {
}

impl<T: Scalar, At: Mat<T>,
     AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>,
     S: GemmNode<T, At, CiT, Ct>>
    GemmNode<T, At, Subcomputation<T, AiT, BiT, CiT>, Ct>
    for ForceB<T, At, AiT, BiT, CiT, Ct, Sc, S> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At,
                  b: &mut Subcomputation<T, AiT, BiT, CiT>,
                  c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        self.subalgorithm.run(&mut b.a, &mut b.b, &mut b.c, thr);
        let new_b = &mut b.c;
        self.child.run(a, new_b, c, thr);
    }

    fn new() -> Self {
        ForceB {child: S::new(), subalgorithm: Sc::new(),
                _t: PhantomData, _ait: PhantomData,
                _bit: PhantomData, _cit: PhantomData,
                _at: PhantomData, _ct: PhantomData }
    }

    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}
