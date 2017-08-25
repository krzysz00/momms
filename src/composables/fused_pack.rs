use core::marker::PhantomData;
use matrix::{Scalar,Mat,PackPair,ResizableBuffer};
//use typenum::Unsigned;
use thread_comm::ThreadInfo;
use composables::{Gemm3Node,AlgorithmStep};

pub struct DelayedPackA<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>,
                        Xt: Mat<T>, Apt: Mat<T>,
                        S: Gemm3Node<T, PackPair<T,At,Apt>, Bt, Ct, Xt>> {
    child: S,
    a_pack: Apt,
    algo_desc: Vec<AlgorithmStep>,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _xt: PhantomData<Xt>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
     Apt: Mat<T>, S> DelayedPackA<T, At, Bt, Ct, Xt, Apt, S>
    where Apt: ResizableBuffer<T>,
          S: Gemm3Node<T, PackPair<T, At, Apt>, Bt, Ct, Xt> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>, Apt: Mat<T>, S>
    Gemm3Node<T, At, Bt, Ct, Xt> for DelayedPackA<T, At, Bt, Ct, Xt, Apt, S>
    where Apt: ResizableBuffer<T>,
          S: Gemm3Node<T, PackPair<T, At, Apt>, Bt, Ct, Xt> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, x: &mut Xt,
                  thr: &ThreadInfo<T>) -> () {
        let y_marker = AlgorithmStep::M{bsz: 0};
        let x_marker = AlgorithmStep::K{bsz: 0};

        let capacity_for_apt = Apt:: capacity_for(a, y_marker, x_marker, &self.algo_desc);
        thr.barrier();

        //Check if we need to resize packing buffer
        if self.a_pack.capacity() < capacity_for_apt {
            if thr.thread_id() == 0 {
                self.a_pack.aquire_buffer_for(capacity_for_apt);
            }
            else {
                self.a_pack.set_capacity(capacity_for_apt);
            }
            self.a_pack.send_alias(thr);
        }

        //Logically resize the a_pack matrix
        self.a_pack.resize_to(a, y_marker, x_marker, &self.algo_desc);
        let mut pair = PackPair::new(a.make_alias(), self.a_pack.make_alias());
        self.child.run(&mut pair, b, c, x, thr);
    }
    fn new() -> Self {
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::M{bsz: 0};
        let x_marker = AlgorithmStep::K{bsz: 0};

        DelayedPackA{ child: S::new(),
               a_pack: Apt::empty(y_marker, x_marker, &algo_desc), algo_desc: algo_desc,
                      _t: PhantomData, _at: PhantomData,
                      _bt: PhantomData, _ct: PhantomData,
                      _xt: PhantomData, }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}

pub struct DelayedPackB<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>,
                        Xt: Mat<T>, Bpt: Mat<T>,
                        S: Gemm3Node<T, At, PackPair<T, Bt, Bpt>, Ct, Xt>> {
    child: S,
    b_pack: Bpt,
    algo_desc: Vec<AlgorithmStep>,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _xt: PhantomData<Xt>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
     Bpt: Mat<T>, S> DelayedPackB<T, At, Bt, Ct, Xt, Bpt, S>
    where Bpt: ResizableBuffer<T>,
          S: Gemm3Node<T, At, PackPair<T, Bt, Bpt>, Ct, Xt> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>, Bpt: Mat<T>, S>
    Gemm3Node<T, At, Bt, Ct, Xt> for DelayedPackB<T, At, Bt, Ct, Xt, Bpt, S>
    where Bpt: ResizableBuffer<T>,
          S: Gemm3Node<T, At, PackPair<T, Bt, Bpt>, Ct, Xt> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, x: &mut Xt,
                  thr: &ThreadInfo<T>) -> () {
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};
        let capacity_for_bpt = Bpt:: capacity_for(b, y_marker, x_marker, &self.algo_desc);

        thr.barrier();

        //Check if we need to resize packing buffer
        if self.b_pack.capacity() < capacity_for_bpt {
            if thr.thread_id() == 0 {
                self.b_pack.aquire_buffer_for(capacity_for_bpt);
            }
            else {
                self.b_pack.set_capacity(capacity_for_bpt);
            }
            self.b_pack.send_alias(thr);
        }

        //Logically resize the c_pack matrix
        self.b_pack.resize_to(b, y_marker, x_marker, &self.algo_desc);
        let mut pair = PackPair::new(b.make_alias(), self.b_pack.make_alias());
        self.child.run(a, &mut pair, c, x, thr);
    }
    fn new() -> Self {
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};

        DelayedPackB { child: S::new(),
                       b_pack: Bpt::empty(y_marker, x_marker, &algo_desc),
                       algo_desc: algo_desc,
                       _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData,
                       _xt: PhantomData,
        }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}

pub struct UnpairA<T: Scalar, At: Mat<T>, Apt: Mat<T>, Bt: Mat<T>,
                   Ct: Mat<T>, Xt: Mat<T>,
                   S: Gemm3Node<T, Apt, Bt, Ct, Xt>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _apt: PhantomData<Apt>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _xt: PhantomData<Xt>,
}
impl<T: Scalar, At: Mat<T>, Apt: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>, S>
    Gemm3Node<T, PackPair<T, At, Apt>, Bt, Ct, Xt> for UnpairA<T, At, Apt, Bt, Ct, Xt, S>
    where S: Gemm3Node<T, Apt, Bt, Ct, Xt> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut PackPair<T, At, Apt>, b: &mut Bt, c: &mut Ct, x: &mut Xt,
                  thr: &ThreadInfo<T>) -> () {
        self.child.run( &mut a.ap, b, c, x, thr);
    }
    fn new() -> Self {
        UnpairA {child: S::new(),
                 _t: PhantomData, _at: PhantomData, _apt: PhantomData,
                 _bt: PhantomData, _ct: PhantomData, _xt: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}

pub struct UnpairB<T: Scalar, At: Mat<T>, Bt: Mat<T>, Bpt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>,
                   S: Gemm3Node<T, At, Bpt, Ct, Xt>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _bpt: PhantomData<Bpt>,
    _ct: PhantomData<Ct>,
    _xt: PhantomData<Xt>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Bpt: Mat<T>, Ct: Mat<T>, Xt: Mat<T>, S>
    Gemm3Node<T, At, PackPair<T, Bt, Bpt>, Ct, Xt> for UnpairB<T, At, Bt, Bpt, Ct, Xt, S>
    where S: Gemm3Node<T, At, Bpt, Ct, Xt> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut PackPair<T, Bt, Bpt>, c: &mut Ct,
                  x: &mut Xt, thr: &ThreadInfo<T>) -> () {
        self.child.run( a, &mut b.ap, c, x, thr);
    }
    fn new() -> Self {
        UnpairB { child: S::new(),
                  _t: PhantomData, _at: PhantomData, _bt: PhantomData,
                  _bpt: PhantomData, _ct: PhantomData, _xt: PhantomData, }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}

pub struct UnpairC<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Cpt: Mat<T>, Xt: Mat<T>,
                   S: Gemm3Node<T, At, Bt, Cpt, Xt>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _cpt: PhantomData<Cpt>,
    _xt: PhantomData<Xt>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Cpt: Mat<T>, Xt: Mat<T>, S>
    Gemm3Node<T, At, Bt, PackPair<T, Ct, Cpt>, Xt> for UnpairC<T, At, Bt, Ct, Cpt, Xt, S>
    where S: Gemm3Node<T, At, Bt, Cpt, Xt> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut PackPair<T, Ct, Cpt>,
                  x: &mut Xt, thr: &ThreadInfo<T>) -> () {
        self.child.run( a, b, &mut c.ap, x, thr);
    }
    fn new() -> Self {
        UnpairC{ child: S::new(),
                 _t: PhantomData, _at: PhantomData,
                 _bt: PhantomData, _ct: PhantomData, _cpt: PhantomData,
                 _xt: PhantomData, }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}
