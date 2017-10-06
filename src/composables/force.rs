use matrix::{Scalar,Mat,Subcomputation,ResizableBuffer};
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};
use std::marker::PhantomData;
use std::convert::AsMut;

// TODO: remove on non-lexical lifetimes?
// Used to prevent rather understandable borrowcheck complaints on resizing
struct MetadataOnlyMatrix<T: Scalar> {
    iter_height: usize,
    iter_width: usize,
    logical_w_padding: usize,
    logical_h_padding: usize,
    width: usize,
    height: usize,
    _t: PhantomData<T>,
}

impl<T: Scalar> MetadataOnlyMatrix<T> {
    pub fn new(other: &Mat<T>) -> Self{
        MetadataOnlyMatrix {
            iter_width: other.iter_width(),
            iter_height: other.iter_height(),
            logical_w_padding: other.logical_w_padding(),
            logical_h_padding: other.logical_h_padding(),
            width: other.width(),
            height: other.height(),
            _t: PhantomData,
        }
    }
}

impl<T: Scalar> Mat<T> for MetadataOnlyMatrix<T> {
    #[inline(always)]
    fn get(&self, _y: usize, _x: usize) -> T { T::zero() }
    #[inline(always)]
    fn set(&mut self, _y: usize, _x: usize, _alpha: T) { }

    #[inline(always)]
    fn iter_height(&self) -> usize {
        self.iter_height
    }
    #[inline(always)]
    fn iter_width(&self) -> usize {
        self.iter_width
    }
    #[inline(always)]
    fn logical_h_padding(&self) -> usize {
        self.logical_h_padding
    }
    #[inline(always)]
    fn logical_w_padding(&self) -> usize {
        self.logical_w_padding
    }
    #[inline(always)]
    fn height(&self) -> usize {
        self.height
    }
    #[inline(always)]
    fn width(&self) -> usize {
        self.width
    }

    #[inline(always)]
    fn set_scalar(&mut self, _alpha: T) { }
    #[inline(always)]
    fn get_scalar(&self) -> T { T::zero() }
    fn push_y_split(&mut self, _start: usize, _end: usize) { }
    fn push_x_split(&mut self, _start: usize, _end: usize) { }
    #[inline(always)]
    fn pop_y_split(&mut self) { }
    #[inline(always)]
    fn pop_x_split(&mut self) { }
    fn push_y_view(&mut self, _blksz: usize) -> usize { 0 }
    fn push_x_view(&mut self, _blksz: usize) -> usize { 0 }
    #[inline(always)]
    fn pop_y_view(&mut self) { }
    #[inline(always)]
    fn pop_x_view(&mut self) { }
    fn slide_y_view_to(&mut self, _y: usize, _blksz: usize) { }
    fn slide_x_view_to(&mut self, _x: usize, _blksz: usize) {}
    #[inline(always)]
    unsafe fn make_alias(&self) -> Self { unimplemented!(); }
    #[inline(always)]
    unsafe fn send_alias(&mut self, _thr: &ThreadInfo<T>) { unimplemented!(); }
}

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
