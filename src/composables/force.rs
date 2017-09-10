use matrix::{Scalar,Mat,Subcomputation,ResizableBuffer};
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};
use std::marker::PhantomData;

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
    // These set outputs of C
    #[inline(always)]
    fn get(&self, _y: usize, _x: usize) -> T { unimplemented!(); }
    #[inline(always)]
    fn set(&mut self, _y: usize, _x: usize, _alpha: T) { unimplemented!(); }

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
    fn set_scalar(&mut self, _alpha: T) { unimplemented!(); }
    #[inline(always)]
    fn get_scalar(&self) -> T { unimplemented!(); }
    fn push_y_split(&mut self, _start: usize, _end: usize) { unimplemented!(); }
    fn push_x_split(&mut self, _start: usize, _end: usize) { unimplemented!(); }
    #[inline(always)]
    fn pop_y_split(&mut self) { unimplemented!(); }
    #[inline(always)]
    fn pop_x_split(&mut self) { unimplemented!(); }
    fn push_y_view(&mut self, _blksz: usize) -> usize { unimplemented!(); }
    fn push_x_view(&mut self, _blksz: usize) -> usize { unimplemented!(); }
    #[inline(always)]
    fn pop_y_view(&mut self) { unimplemented!(); }
    #[inline(always)]
    fn pop_x_view(&mut self) { unimplemented!(); }
    fn slide_y_view_to(&mut self, _y: usize, _blksz: usize) { unimplemented!(); }
    fn slide_x_view_to(&mut self, _x: usize, _blksz: usize) { unimplemented!(); }
    #[inline(always)]
    unsafe fn make_alias(&self) -> Self { unimplemented!(); }
    #[inline(always)]
    unsafe fn send_alias(&mut self, thr: &ThreadInfo<T>) { unimplemented!(); }
}

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
    for ForceA<T, AiT, BiT, CiT, Bt, Ct, Sc, S> {
    #[inline(always)]
    default unsafe fn run(&mut self, a: &mut Subcomputation<T, AiT, BiT, CiT>,
                          b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
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
                _bt: PhantomData, _ct: PhantomData }
    }

    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}

impl<T: Scalar, AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     Bt: Mat<T>, Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>,
     S: GemmNode<T, CiT, Bt, Ct>>
    GemmNode<T, Subcomputation<T, AiT, BiT, CiT>, Bt, Ct>
    for ForceA<T, AiT, BiT, CiT, Bt, Ct, Sc, S>
    where CiT: ResizableBuffer<T> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut Subcomputation<T, AiT, BiT, CiT>,
                  b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let dummy_algo_desc: [AlgorithmStep; 0] = [];
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};
        // Here, a has height from a.a and width from a.b
        let metadata = MetadataOnlyMatrix::new(a);
        let capacity_for_cit = CiT:: capacity_for(&metadata, y_marker, x_marker, &dummy_algo_desc);

        //Check if we need to resize packing buffer
        if a.c.capacity() < capacity_for_cit {
            if thr.thread_id() == 0 {
                a.c.aquire_buffer_for(capacity_for_cit);
            }
            else {
                a.c.set_capacity(capacity_for_cit);
            }
            //self.b_pack.send_alias(thr);
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
    default unsafe fn run(&mut self, a: &mut At,
                  b: &mut Subcomputation<T, AiT, BiT, CiT>,
                  c: &mut Ct, thr: &ThreadInfo<T>) -> () {
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
                _at: PhantomData, _ct: PhantomData }
    }

    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}

impl<T: Scalar, At: Mat<T>,
     AiT: Mat<T>, BiT: Mat<T>, CiT: Mat<T>,
     Ct: Mat<T>,
     Sc: GemmNode<T, AiT, BiT, CiT>,
     S: GemmNode<T, At, CiT, Ct>>
    GemmNode<T, At, Subcomputation<T, AiT, BiT, CiT>, Ct>
    for ForceB<T, At, AiT, BiT, CiT, Ct, Sc, S>
    where CiT: ResizableBuffer<T> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At,
                  b: &mut Subcomputation<T, AiT, BiT, CiT>,
                  c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let dummy_algo_desc: [AlgorithmStep; 0] = [];
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};
        // Here, b has height from b.a and width from b.b
        let metadata = MetadataOnlyMatrix::new(b);
        let capacity_for_cit = CiT:: capacity_for(&metadata, y_marker, x_marker, &dummy_algo_desc);

        //Check if we need to resize packing buffer
        if b.c.capacity() < capacity_for_cit {
            if thr.thread_id() == 0 {
                b.c.aquire_buffer_for(capacity_for_cit);
            }
            else {
                b.c.set_capacity(capacity_for_cit);
            }
            //self.b_pack.send_alias(thr);
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
