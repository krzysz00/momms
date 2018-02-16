use matrix::{Scalar,Mat,Subcomputation,ResizableBuffer,MetadataOnlyMatrix};
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};
use std::marker::PhantomData;
use std::convert::AsMut;

pub struct ForceA<T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
                  At: AsMut<Subcomputation<T, AiT, BiT, CiT>>,
                  Bt: Mat<T>, Ct: Mat<T>,
                  Sc: GemmNode<T, AiT, BiT, CiT>,
                  S: GemmNode<T, CiT, Bt, Ct>> {
    child: S,
    subalgorithm: Sc,
    _t: PhantomData<T>,
    _ait: PhantomData<AiT>,
    _bit: PhantomData<BiT>,
    _cit: PhantomData<CiT>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}

impl<T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     At: AsMut<Subcomputation<T, AiT, BiT, CiT>>,
     Bt: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>, S: GemmNode<T, CiT, Bt, Ct>>
    ForceA<T, AiT, BiT, CiT, At, Bt, Ct, Sc, S> {
}

impl<T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     At: AsMut<Subcomputation<T, AiT, BiT, CiT>>,
     Bt: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>,
     S: GemmNode<T, CiT, Bt, Ct>>
    GemmNode<T, At, Bt, Ct>
    for ForceA<T, AiT, BiT, CiT, At, Bt, Ct, Sc, S>
    where At: Mat<T> {
    #[inline(always)]
    default unsafe fn run(&mut self, a: &mut At,
                          b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let a: &mut Subcomputation<T, AiT, BiT, CiT> = a.as_mut();
        let beta_save = a.c.get_scalar();
        a.c.set_scalar(T::zero());
        self.subalgorithm.run(&mut a.a, &mut a.b, &mut a.c, thr);
        a.c.set_scalar(beta_save);
        let new_a = &mut a.c;
        self.child.run(new_a, b, c, thr);
    }

    fn new() -> Self {
        ForceA {child: S::new(), subalgorithm: Sc::new(),
                _t: PhantomData, _ait: PhantomData,
                _bit: PhantomData, _cit: PhantomData,
                _at: PhantomData,
                _bt: PhantomData, _ct: PhantomData }
    }

    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}

impl<T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     At: AsMut<Subcomputation<T, AiT, BiT, CiT>>,
     Bt: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>,
     S: GemmNode<T, CiT, Bt, Ct>>
    GemmNode<T, At, Bt, Ct>
    for ForceA<T, AiT, BiT, CiT, At, Bt, Ct, Sc, S>
    where CiT: ResizableBuffer<T>,
          At: Mat<T> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At,
                  b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let dummy_algo_desc: [AlgorithmStep; 0] = [];
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};
        let metadata = MetadataOnlyMatrix::new(a);
        let a: &mut Subcomputation<T, AiT, BiT, CiT> = a.as_mut();
        // Here, a has height from a.a and width from a.b
        let capacity_for_cit = CiT:: capacity_for(&metadata, y_marker, x_marker, &dummy_algo_desc);

        //Check if we need to resize packing buffer
        if a.c.capacity() < capacity_for_cit {
            if thr.thread_id() == 0 {
                a.c.aquire_buffer_for(capacity_for_cit);
            }
            else {
                a.c.set_capacity(capacity_for_cit);
            }
            a.c.send_alias(thr);
        }

        //Logically resize the output matrix
        a.c.resize_to(&metadata, y_marker, x_marker, &dummy_algo_desc);

        let beta_save = a.c.get_scalar();
        a.c.set_scalar(T::zero());

        self.subalgorithm.run(&mut a.a, &mut a.b, &mut a.c, thr);
        a.c.set_scalar(beta_save);

        let new_a = &mut a.c;
        self.child.run(new_a, b, c, thr);
    }
}

pub struct ForceB<T: Scalar, At: Mat<T>,
                  AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
                  Bt: AsMut<Subcomputation<T, AiT, BiT, CiT>>,
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
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}

impl<T: Scalar, At: Mat<T>,
     AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     Bt: AsMut<Subcomputation<T, AiT, BiT, CiT>>,
     Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>, S: GemmNode<T, At, CiT, Ct>>
    ForceB<T, At, AiT, BiT, CiT, Bt, Ct, Sc, S> {
}

impl<T: Scalar, At: Mat<T>,
     AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     Bt: AsMut<Subcomputation<T, AiT, BiT, CiT>>,
     Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>,
     S: GemmNode<T, At, CiT, Ct>>
    GemmNode<T, At, Bt, Ct>
    for ForceB<T, At, AiT, BiT, CiT, Bt, Ct, Sc, S>
    where Bt: Mat<T> {
    #[inline(always)]
    default unsafe fn run(&mut self, a: &mut At,
                          b: &mut Bt,
                          c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let b: &mut Subcomputation<T, AiT, BiT, CiT> = b.as_mut();
        let beta_save = b.c.get_scalar();
        b.c.set_scalar(T::zero());
        self.subalgorithm.run(&mut b.a, &mut b.b, &mut b.c, thr);
        b.c.set_scalar(beta_save);
        let new_b = &mut b.c;
        self.child.run(a, new_b, c, thr);
    }

    fn new() -> Self {
        ForceB {child: S::new(), subalgorithm: Sc::new(),
                _t: PhantomData, _ait: PhantomData,
                _bit: PhantomData, _cit: PhantomData,
                _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }

    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}

impl<T: Scalar, At: Mat<T>,
     AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     Bt: AsMut<Subcomputation<T, AiT, BiT, CiT>>,
     Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>,
     S: GemmNode<T, At, CiT, Ct>>
    GemmNode<T, At, Bt, Ct>
    for ForceB<T, At, AiT, BiT, CiT, Bt, Ct, Sc, S>
    where CiT: ResizableBuffer<T>,
          Bt: Mat<T> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At,
                  b: &mut Bt,
                  c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let dummy_algo_desc: [AlgorithmStep; 0] = [];
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};
        let metadata = MetadataOnlyMatrix::new(b);
        let b: &mut Subcomputation<T, AiT, BiT, CiT> = b.as_mut();
        // Here, b has height from b.a and width from b.b
        let capacity_for_cit = CiT:: capacity_for(&metadata, y_marker, x_marker, &dummy_algo_desc);

        //Check if we need to resize packing buffer
        if b.c.capacity() < capacity_for_cit {
            if thr.thread_id() == 0 {
                b.c.aquire_buffer_for(capacity_for_cit);
            }
            else {
                b.c.set_capacity(capacity_for_cit);
            }
            b.c.send_alias(thr);
        }

        //Logically resize the output matrix
        b.c.resize_to(&metadata, y_marker, x_marker, &dummy_algo_desc);

        let beta_save = b.c.get_scalar();
        b.c.set_scalar(T::zero());

        self.subalgorithm.run(&mut b.a, &mut b.b, &mut b.c, thr);
        b.c.set_scalar(beta_save);

        let new_b = &mut b.c;
        self.child.run(a, new_b, c, thr);
    }
}
