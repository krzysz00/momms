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
        let ap = a.get_buffer();
        let bp = b.get_buffer();
        let cp = c.get_mut_buffer();

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

fn flush_cache(arr: &mut Vec<f64> ) {
    for i in (arr).iter_mut() {
        *i += 1.0;
    }
}

fn test_gemm() {
    use typenum::{UInt, B0};
    type NC = UInt<UInt<typenum::U1020, B0>, B0>;
    type KC = typenum::U256;
    type MC = typenum::U72;
    type NR = typenum::U8;
    type MR = typenum::U6;

    type Goto<T: Scalar, MTA: Mat<T>, MTB: Mat<T>, MTC: Mat<T>>
        = PartN<T, MTA, MTB, MTC, NC,
          PartK<T, MTA, MTB, MTC, KC,
          PackB<T, MTA, MTB, MTC, ColumnPanelMatrix<T,NR>,
          PartM<T, MTA, ColumnPanelMatrix<T,NR>, MTC, MC,
          PackA<T, MTA, ColumnPanelMatrix<T,NR>, MTC, RowPanelMatrix<T,MR>,
          KernelNM<T, RowPanelMatrix<T,MR>, ColumnPanelMatrix<T,NR>, MTC, NR, MR>>>>>>;

    type GotoPlain = Goto<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>;

    let mut goto: GotoPlain = GotoPlain::new();

    let flusher_len = 32*1024*1024; //256MB
    let mut flusher: Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len {
        flusher.push(0.0);
    }

    //pin_to_core(0);
    for index in 1..64 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_blis: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let size = index * 16;
        let (m, n, k) = (size, size, size);

        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a : Matrix<f64> = Matrix::new(m, k);
            let mut b : Matrix<f64> = Matrix::new(k, n);
            let mut c : Matrix<f64> = Matrix::new(m, n);
            a.fill_rand(); b.fill_rand(); c.fill_zero();

            c.transpose();
            flush_cache(&mut flusher);

            let mut start = Instant::now();
            unsafe {
                goto.run(&mut a, &mut b, &mut c, &ThreadInfo::single_thread() );
            }
            best_time = best_time.min(dur_seconds(start));
            let err = test_c_eq_a_b( &mut a, &mut b, &mut c);
            worst_err = worst_err.max(err);

            c.transpose();
            flush_cache(&mut flusher);

            start = Instant::now();
            blas_dgemm( &mut a, &mut b, &mut c);
            best_time_blis = best_time_blis.min(dur_seconds(start));
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}",
                 m, n, k,
                 gflops(m,n,k,best_time),
                 gflops(m,n,k,best_time_blis),
                 format!("{:e}", worst_err.sqrt()));
    }

    let sum: f64 = flusher.iter().sum();
    println!("Flush value {}", sum);
}
fn main() {
    test_gemm();
}
