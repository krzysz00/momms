extern crate gemm_oxide;
extern crate core;
extern crate typenum;
extern crate hwloc;
extern crate libc;

use std::time::{Instant};
use std::ffi::{CString};
use self::libc::{c_double, int64_t, c_char};
use typenum::{U1};

use gemm_oxide::kern::hsw::{Ukernel, KernelMN, KernelNM, GemvAL1};
pub use gemm_oxide::matrix::{Scalar, Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix, Hierarch};
pub use gemm_oxide::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB, SpawnThreads, ParallelM, ParallelN, Nwayer};
pub use gemm_oxide::thread_comm::ThreadInfo;
pub use gemm_oxide::triple_loop::TripleLoop;


extern{
    fn dgemm_( transa: *const c_char, transb: *const c_char,
               m: *const int64_t, n: *const int64_t, k: *const int64_t,
               alpha: *const c_double, 
               a: *const c_double, lda: *const int64_t,
               b: *const c_double, ldb: *const int64_t,
               beta: *const c_double,
               c: *mut c_double, ldc: *const int64_t );
}

fn blas_dgemm( a: &mut Matrix<f64>, b: &mut Matrix<f64>, c: &mut Matrix<f64> ) 
{
    unsafe{ 
        let transa = CString::new("N").unwrap();
        let transb = CString::new("N").unwrap();
        let ap = a.get_mut_buffer();
        let bp = b.get_buffer();
        let cp = c.get_buffer();

        let lda = a.get_column_stride() as int64_t;
        let ldb = b.get_column_stride() as int64_t;
        let ldc = c.get_column_stride() as int64_t;

        let m = c.height() as int64_t;
        let n = b.width() as int64_t;
        let k = a.width() as int64_t;

        let alpha: f64 = 1.0;
        let beta: f64 = 1.0;
    
        dgemm_( transa.as_ptr() as *const c_char, transb.as_ptr() as *const c_char,
                &m, &n, &k,
                &alpha as *const c_double, 
                ap as *const c_double, &lda,
                bp as *const c_double, &ldb,
                &beta as *const c_double,
                cp as *mut c_double, &ldc );
    }
}

fn test_c_eq_a_b<T:Scalar, At:Mat<T>, Bt:Mat<T>, Ct:Mat<T>>( a: &mut At, b: &mut Bt, c: &mut Ct ) -> T {
    let mut ref_gemm : TripleLoop = TripleLoop{};

    let m = c.height();
    let n = b.width();
    let k = a.width();

    let mut w : Matrix<T> = Matrix::new(n, 1);
    let mut bw : Matrix<T> = Matrix::new(k, 1);
    let mut abw : Matrix<T> = Matrix::new(m, 1);
    let mut cw : Matrix<T> = Matrix::new(m, 1);
    w.fill_rand();
    cw.fill_zero();
    bw.fill_zero();
    abw.fill_zero();

    //Do bw = Bw, then abw = A*(Bw)   
    unsafe {
        ref_gemm.run( b, &mut w, &mut bw, &ThreadInfo::single_thread() );
        ref_gemm.run( a, &mut bw, &mut abw, &ThreadInfo::single_thread() );
    }

    //Do cw = Cw
    unsafe {
        ref_gemm.run( c, &mut w, &mut cw, &ThreadInfo::single_thread() );
    }
    
    //Cw -= abw
    cw.axpy( T::zero() - T::one(), &abw );
    cw.frosqr()
}

fn dur_seconds(start: Instant) -> f64 {
    let dur = start.elapsed();
    let time_secs = dur.as_secs() as f64;
    let time_nanos = dur.subsec_nanos() as f64;
    time_nanos / 1E9 + time_secs
}

fn gflops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
    let nflops = (m * n * k) as f64;
    2.0 * nflops / seconds / 1E9
}

fn pin_to_core(core: usize) {
    use self::hwloc::{Topology, CPUBIND_THREAD, ObjectType};
    let mut topo = Topology::new();
    let tid = unsafe { libc::pthread_self() };

    let bind_to = {
        let cores = topo.objects_with_type(&ObjectType::Core).unwrap();
        match cores.get(core) {
            Some(val) => val.cpuset().unwrap(),
            None => panic!("No Core found with id {}", core)
        }
    };
    let _ = topo.set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD);
}


fn compare_gotos() {
    type KC = typenum::U256; 
    type MC = typenum::U72; 
    type NR = typenum::U8;
    type MR = typenum::U6;
    type HierA<T> = Hierarch<T, MR, KC, U1, MR>;
    type HierB<T> = Hierarch<T, KC, NR, NR, U1>;
    type HierC<T> = Hierarch<T, MR, NR, NR, U1>;

    type GotoH<T:Scalar,MTA:Mat<T>,MTB:Mat<T>,MTC:Mat<T>> 
        = PartK<T, MTA, MTB, MTC, KC,
          PartM<T, MTA, MTB, MTC, MC,
          KernelNM<T, MTA, MTB, MTC, NR, MR>>>;

    type Goto<T:Scalar,MTA:Mat<T>,MTB:Mat<T>,MTC:Mat<T>> 
        = PartK<T, MTA, MTB, MTC, KC,
          PackB<T, MTA, MTB, MTC, ColumnPanelMatrix<T,NR>,
          PartM<T, MTA, ColumnPanelMatrix<T,NR>, MTC, MC,
          PackA<T, MTA, ColumnPanelMatrix<T,NR>, MTC, RowPanelMatrix<T,MR>,
          KernelNM<T, RowPanelMatrix<T,MR>, ColumnPanelMatrix<T,NR>, MTC, NR, MR>>>>>;

    type GotoOrig  = Goto<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>;
    type GotoHierC = Goto<f64, Matrix<f64>, Matrix<f64>, HierC<f64>>;
    type GotoHier = GotoH<f64, HierA<f64>, HierB<f64>, HierC<f64>>;

    let mut goto : GotoOrig = GotoOrig::new();
    let mut goto_hier_c : GotoHierC = GotoHierC::new();
    let mut goto_hier : GotoHier = GotoHier::new();
    let algo_desc = GotoHier::hierarchy_description();
    
    pin_to_core(0);

    for index in 1..128 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_blis: f64 = 9999999999.0;
        let mut best_time_2: f64 = 9999999999.0;
        let mut best_time_3: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let mut worst_err_2: f64 = 0.0;
        let mut worst_err_3: f64 = 0.0;
        let size = index*8;
        let (m, n, k) = (size, size, size);


        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a : HierA<f64> = Hierarch::new(m, k, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
            let mut b : HierB<f64> = Hierarch::new(k, n, &algo_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
            let mut c : HierC<f64> = Hierarch::new(m, n, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});
            a.fill_rand(); b.fill_rand(); c.fill_zero();

            let mut a2 : Matrix<f64> = Matrix::new(m, k);
            let mut b2 : Matrix<f64> = Matrix::new(k, n);
            let mut c2 : Matrix<f64> = Matrix::new(m, n);
            a2.fill_rand(); b2.fill_rand(); c2.fill_zero();

            c2.transpose();
            let mut start = Instant::now();
            unsafe{
                goto.run( &mut a2, &mut b2, &mut c2, &ThreadInfo::single_thread() );
            }
            best_time = best_time.min(dur_seconds(start));
            let err = test_c_eq_a_b( &mut a2, &mut b2, &mut c2);
            worst_err = worst_err.max(err);
            c2.transpose();

            start = Instant::now();
            unsafe{
                goto_hier_c.run( &mut a2, &mut b2, &mut c, &ThreadInfo::single_thread() );
            }
            best_time_2 = best_time_2.min(dur_seconds(start));
            let err = test_c_eq_a_b( &mut a2, &mut b2, &mut c);
            worst_err_2 = worst_err_2.max(err);
            
            c.fill_zero();           
            start = Instant::now();
            unsafe{
                goto_hier.run( &mut a, &mut b, &mut c, &ThreadInfo::single_thread() );
            }
            best_time_3 = best_time_3.min(dur_seconds(start));
            let err = test_c_eq_a_b( &mut a, &mut b, &mut c);
            worst_err_3 = worst_err_3.max(err);
            

            start = Instant::now();
            blas_dgemm( &mut a2, &mut b2, &mut c2);
            best_time_blis = best_time_blis.min(dur_seconds(start));
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", 
                 m, n, k,
                 gflops(m,n,k,best_time), 
                 gflops(m,n,k,best_time_2), 
                 gflops(m,n,k,best_time_3), 
                 gflops(m,n,k,best_time_blis),
                 format!("{:e}", worst_err.sqrt()),
                 format!("{:e}", worst_err_2.sqrt()),
                 format!("{:e}", worst_err_3.sqrt()));

    }
}

fn test_gemv_kernel() {
    type KC = typenum::U64; 
    type MC = typenum::U56;
    type NC = typenum::U1024;

    type HierA<T> = Hierarch<T, MC, KC, U1, MC>;
    type HierB<T> = Hierarch<T, KC, NC, U1, KC>;
    type HierC<T> = Hierarch<T, MC, NC, U1, MC>;

    type Algorithm<T:Scalar,MTA:Mat<T>,MTB:Mat<T>,MTC:Mat<T>> 
        = PartK<T, MTA, MTB, MTC, typenum::U128,
          PartN<T, MTA, MTB, MTC, typenum::U192, 
          PartK<T, MTA, MTB, MTC, KC,
          PartM<T, MTA, MTB, MTC, MC,
          GemvAL1<T, MTA, MTB, MTC, MC, KC>>>>>;

    type Algo = Algorithm<f64, HierA<f64>, HierB<f64>, HierC<f64>>;

    let mut algo : Algo = Algo::new();
    let algo_desc = Algo::hierarchy_description();
    
    pin_to_core(0);

    for index in 1..128 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_blis: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let size = index*8;
        let (m, n, k) = (size, size, size);

        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a : HierA<f64> = Hierarch::new(m, k, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
            let mut b : HierB<f64> = Hierarch::new(k, n, &algo_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
            let mut c : HierC<f64> = Hierarch::new(m, n, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});
            a.fill_rand(); b.fill_rand(); c.fill_zero();

            let mut a2 : Matrix<f64> = Matrix::new(m, k);
            let mut b2 : Matrix<f64> = Matrix::new(k, n);
            let mut c2 : Matrix<f64> = Matrix::new(m, n);
            a2.fill_rand(); b2.fill_rand(); c2.fill_zero();

            let mut start = Instant::now();
            unsafe{
                algo.run( &mut a, &mut b, &mut c, &ThreadInfo::single_thread() );
            }
            best_time = best_time.min(dur_seconds(start));
            let err = test_c_eq_a_b( &mut a, &mut b, &mut c);
            worst_err = worst_err.max(err);

/*
            print!("A = ");
            a.print_matlab();
            println!(";");
            println!("");
            print!("B = ");
            b.print_matlab();
            println!(";");
            println!("");
            print!("C = ");
            c.print_matlab();
            println!(";");
            println!("");
            
            println!("");
            c.print();
*/
            start = Instant::now();
            blas_dgemm( &mut a2, &mut b2, &mut c2);
            best_time_blis = best_time_blis.min(dur_seconds(start));
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}", 
                 m, n, k,
                 gflops(m,n,k,best_time), 
                 gflops(m,n,k,best_time_blis),
                 format!("{:e}", worst_err.sqrt()));
    }
}

fn main() {
    test_gemv_kernel();
//    time_sweep_goto( );
}