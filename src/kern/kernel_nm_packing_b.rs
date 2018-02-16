use matrix::{Scalar,Mat,RoCM,Matrix,
             ResizableBuffer,ColumnPanelMatrix,MetadataOnlyMatrix};
use core::ptr;
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::Unsigned;
use super::ukernel_wrapper::{UkernelWrapper,GenericUkernelWrapper};

#[inline(always)]
unsafe fn prefetch_c_row<T: Scalar>(ptr: *mut T) {
    asm!(" 
         prefetchw ($0)
         prefetchw 64($0)
         " 
         : : "r"(ptr));
}

pub struct KernelNMPackingB<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Nr: Unsigned, Mr: Unsigned> {
    tmp: Matrix<T>,
    b_pack: ColumnPanelMatrix<T, Nr>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _nrt: PhantomData<Nr>,
    _mrt: PhantomData<Mr>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Nr: Unsigned, Mr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for KernelNMPackingB<T, At, Bt, Ct, Nr, Mr> 
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>
{
    #[inline(always)]
    default unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        //A must be column major
        debug_assert!(a.get_leaf_rs() == 1 && a.get_leaf_cs() == Mr::to_usize());

        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha = a.get_scalar() * b.get_scalar();
        let mut beta = c.get_scalar();

        let c_leaf_rs = c.get_leaf_rs() as isize;
        let c_leaf_cs = c.get_leaf_cs() as isize;

        let b_rs = b.get_block_rs(1, 1) as isize;
        let b_cs = b.get_block_cs(1, 1) as isize;

        let c_nr_stride = c.get_block_cs(1, Nr::to_usize()) as isize;
        let b_nr_stride = b.get_block_cs(1, Nr::to_usize()) as isize;

        let c_mr_stride = c.get_block_rs(1, Mr::to_usize()) as isize;
        let a_mr_stride = a.get_block_rs(1, Mr::to_usize()) as isize;

        let mut c_jr = cp;
        let mut b_jr = bp;

        let dummy_algo_desc: [AlgorithmStep; 0] = [];
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};
        let metadata = MetadataOnlyMatrix::new_from_data(Nr::to_usize(), k as usize,
                                                         0, 0,
                                                         Nr::to_usize(), k as usize);
        let capacity_for_bpack = ColumnPanelMatrix::<T, Nr>
            ::capacity_for(&metadata, y_marker, x_marker, &dummy_algo_desc);

        if self.b_pack.capacity() < capacity_for_bpack {
            if thr.thread_id() == 0 {
                self.b_pack.aquire_buffer_for(capacity_for_bpack);
            }
            else {
                self.b_pack.set_capacity(capacity_for_bpack);
            }
            self.b_pack.send_alias(thr);
        }

        //Logically resize the output matrix
        self.b_pack.resize_to(&metadata, y_marker, x_marker, &dummy_algo_desc);
        let bpack_ptr = self.b_pack.get_mut_buffer();

        let mut jr: isize = 0;

        while jr < n {
            b.establish_leaf(0, (jr as usize) / Nr::to_usize(), k as usize, Nr::to_usize());

            let mut bpack_pack_p = bpack_ptr;
            let mut bp_pack_p = b_jr;
            for _ in 0..k {
                for i in 0..Nr::to_isize() {
                    ptr::write(bpack_pack_p.offset(i), ptr::read(bp_pack_p.offset(i * b_cs)));
                }
                bpack_pack_p = bpack_pack_p.offset(Nr::to_isize());
                bp_pack_p = bp_pack_p.offset(b_rs);
            }

            let mut ir : isize = 0;
            let mut a_ir = ap;
            let mut c_ir = c_jr;
            while ir < m {
                //prefetch next C
                //These prefetches are only correct if C is row major!!!!
                if cfg!(feature="asm_snippets") {
                    let next_c_ir = c_ir.offset(c_mr_stride);
                    prefetch_c_row(next_c_ir); 
                    prefetch_c_row(next_c_ir.offset(c_leaf_rs)); 
                    prefetch_c_row(next_c_ir.offset(2*c_leaf_rs)); 
                    prefetch_c_row(next_c_ir.offset(3*c_leaf_rs)); 
                }

                if Ct::full_leaves() || (n - jr >= Nr::to_isize()) && (m - ir >= Mr::to_isize()) {
                    <UkernelWrapper<Mr, Nr, T>>::run(k, &mut alpha, a_ir, bpack_ptr, &mut beta, c_ir, c_leaf_rs, c_leaf_cs);
                } else {
					let tp = self.tmp.get_mut_buffer();
					let mut t_scalar = T::zero();
                    let t_rs = self.tmp.get_row_stride() as isize;
                    let t_cs = self.tmp.get_column_stride() as isize;
					<UkernelWrapper<Mr,Nr,T>>::run(k, &mut alpha, a_ir, bpack_ptr, &mut t_scalar, tp, t_rs, t_cs);

                    let local_m = if m-ir >= Mr::to_isize() { Mr::to_isize() } else { m-ir };
                    let local_n = if n-jr >= Nr::to_isize() { Nr::to_isize() } else { n-jr };

					//Add t to c
                    for ii in 0..local_m {
                        for jj in 0..local_n {
                            let tau = ptr::read(tp.offset(ii * t_rs + jj * t_cs));
                            let chi = ptr::read(c_ir.offset(ii * c_leaf_rs + jj * c_leaf_cs));
                            ptr::write(c_ir.offset(ii * c_leaf_rs + jj * c_leaf_cs), tau+beta*chi);
                        }
                    }
                }

                ir += Mr::to_isize();
                a_ir = a_ir.offset(a_mr_stride);
                c_ir = c_ir.offset(c_mr_stride);
            }
            jr += Nr::to_isize();
            c_jr = c_jr.offset(c_nr_stride);
            b_jr = b_jr.offset(b_nr_stride);
        }

    }
    fn new() -> Self {
        let mut tmp = <Matrix<T>>::new(Nr::to_usize(), Mr::to_usize());
        tmp.transpose();
        let b_pack = <ColumnPanelMatrix<T, Nr>>::new(Nr::to_usize(), 256);
        KernelNMPackingB{ tmp: tmp, b_pack: b_pack,
                          _at: PhantomData, _bt: PhantomData,
                          _ct: PhantomData, _nrt: PhantomData, _mrt: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut desc = Vec::new();
        desc.push(AlgorithmStep::M{bsz: Mr::to_usize()});
        desc.push(AlgorithmStep::N{bsz: Nr::to_usize()});
        desc
    }
}
