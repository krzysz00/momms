use matrix::{Scalar,Mat,Subcomputation};
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};
use std::marker::PhantomData;

pub struct ForceA<T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
                  Bt: Mat<T>, Ct: Mat<T>,
                  Sc: GemmNode<T, AiT, BiT, CiT>,
                  S: GemmNode<T, CiT, Bt, Ct>>
    where Sc: Send {
    child: S,

    _t: PhantomData<T>,
    _ait: PhantomData<AiT>,
    _bit: PhantomData<BiT>,
    _cit: PhantomData<CiT>,
    _sc: PhantomData<Sc>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}

impl<T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>, Bt: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>, S: GemmNode<T, CiT, Bt, Ct>>
    ForceA<T, AiT, BiT, CiT, Bt, Ct, Sc, S> where Sc: Send {
}

impl<'a, T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     Bt: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>,
     S: GemmNode<T, CiT, Bt, Ct>>
    GemmNode<T, Subcomputation<'a, T, AiT, BiT, CiT, Sc>, Bt, Ct>
    for ForceA<T, AiT, BiT, CiT, Bt, Ct, Sc, S> where Sc: Send
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut Subcomputation<T, AiT, BiT, CiT, Sc>,
                  b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let new_a = a.force(thr);
        self.child.run(new_a, b, c, thr);
    }

    fn new() -> Self {
        ForceA {child: S::new(), _t: PhantomData,
                _ait: PhantomData, _bit: PhantomData,
                _cit: PhantomData, _sc: PhantomData,
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
                  S: GemmNode<T, At, CiT, Ct>>
    where Sc: Send {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _ait: PhantomData<AiT>,
    _bit: PhantomData<BiT>,
    _cit: PhantomData<CiT>,
    _sc: PhantomData<Sc>,
    _ct: PhantomData<Ct>,
}

impl<T: Scalar, At: Mat<T>,
     AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>, S: GemmNode<T, At, CiT, Ct>>
    ForceB<T, At, AiT, BiT, CiT, Ct, Sc, S> where Sc: Send {
}

impl<'a, T: Scalar, At: Mat<T>,
     AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>,
     S: GemmNode<T, At, CiT, Ct>>
    GemmNode<T, At, Subcomputation<'a, T, AiT, BiT, CiT, Sc>, Ct>
    for ForceB<T, At, AiT, BiT, CiT, Ct, Sc, S> where Sc: Send
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At,
                  b: &mut Subcomputation<'a, T, AiT, BiT, CiT, Sc>,
                  c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let new_b = b.force(thr);
        self.child.run(a, new_b, c, thr);
    }

    fn new() -> Self {
        ForceB {child: S::new(), _t: PhantomData,
                _ait: PhantomData, _bit: PhantomData,
                _cit: PhantomData, _sc: PhantomData,
                _at: PhantomData, _ct: PhantomData }
    }

    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}
