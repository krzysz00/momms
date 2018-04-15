extern crate hwloc;
extern crate libc;
extern crate alloc;

use std::time::Instant;
use std::cmp;
#[allow(unused_imports)]
use libc::{c_double, int64_t, c_char};

#[allow(unused_imports)]
use std::ffi::{CString};
use thread_comm::ThreadInfo;
#[allow(unused_imports)]
use matrix::{Scalar, Mat, Matrix, RoCM};
use composables::{GemmNode, TripleLoop};
use self::alloc::heap::Layout;

#[cfg(feature="blis")]
extern{
    fn dgemm_( transa: *const c_char, transb: *const c_char,
               m: *const int64_t, n: *const int64_t, k: *const int64_t,
               alpha: *const c_double, 
               a: *const c_double, lda: *const int64_t,
               b: *const c_double, ldb: *const int64_t,
               beta: *const c_double,
               c: *mut c_double, ldc: *const int64_t );
}

#[cfg(feature="blis")]
pub fn blas_dgemm( a: &mut Matrix<f64>, b: &mut Matrix<f64>, c: &mut Matrix<f64> ) 
{
    unsafe {
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

#[cfg(feature="blis")]
pub fn blis_dgemm(a: &mut Matrix<f64>, b: &mut Matrix<f64>, c: &mut Matrix<f64>) {
    use blis_types::{bli_dgemm,dim_t,inc_t,trans_t};
    let m = c.height() as dim_t;
    let n = c.width() as dim_t;
    let k = a.width() as dim_t;

    let rs_a = a.get_row_stride() as inc_t;
    let cs_a = a.get_column_stride() as inc_t;
    let rs_b = b.get_row_stride() as inc_t;
    let cs_b = b.get_column_stride() as inc_t;
    let rs_c = c.get_row_stride() as inc_t;
    let cs_c = c.get_column_stride() as inc_t;

    let mut alpha = a.get_scalar() * b.get_scalar();
    let mut beta = c.get_scalar();

    unsafe {
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        bli_dgemm(trans_t::BLIS_NO_TRANSPOSE, trans_t::BLIS_NO_TRANSPOSE,
                  m, n, k,
                  &mut alpha as *mut f64,
                  ap as *mut f64, rs_a, cs_a,
                  bp as *mut f64, rs_b, cs_b,
                  &mut beta as *mut f64,
                  cp as *mut f64, rs_c, cs_c,
                  ::std::ptr::null_mut());
    }
}

pub fn test_c_eq_a_b<T:Scalar, At:Mat<T>, Bt:Mat<T>, Ct:Mat<T>>( a: &mut At, b: &mut Bt, c: &mut Ct ) -> T {
    let mut ref_gemm: TripleLoop = TripleLoop{};

    let m = c.height();
    let n = b.width();
    let k = a.width();

    let mut w: Matrix<T> = Matrix::new(n, 1);
    let mut bw: Matrix<T> = Matrix::new(k, 1);
    let mut abw: Matrix<T> = Matrix::new(m, 1);
    let mut cw: Matrix<T> = Matrix::new(m, 1);
    w.fill_rand();
    cw.fill_zero();
    bw.fill_zero();
    abw.fill_zero();

    //Do bw = Bw, then abw = A*(Bw)
    unsafe {
        ref_gemm.run(b, &mut w, &mut bw, &ThreadInfo::single_thread() );
        ref_gemm.run(a, &mut bw, &mut abw, &ThreadInfo::single_thread() );
    }

    //Do cw = Cw
    unsafe {
        ref_gemm.run( c, &mut w, &mut cw, &ThreadInfo::single_thread() );
    }

    //Cw -= abw
    cw.axpy( T::zero() - T::one(), &abw );
    cw.frosqr()
}

pub fn test_d_eq_a_b_c<T:Scalar, At: Mat<T>, Bt: Mat<T>,
                   Ct: Mat<T>, Dt: Mat<T>>(a: &mut At, b: &mut Bt, c: &mut Ct, d: &mut Dt) -> T {
    let mut ref_gemm : TripleLoop = TripleLoop{};

    let m = d.height();
    let n = c.width();
    let l = b.width();
    let k = a.width();

    let mut w: Matrix<T> = Matrix::new(n, 1);
    let mut cw: Matrix<T> = Matrix::new(l, 1);
    let mut bcw: Matrix<T> = Matrix::new(k, 1);
    let mut abcw: Matrix<T> = Matrix::new(m, 1);
    let mut dw: Matrix<T> = Matrix::new(m, 1);
    w.fill_rand();
    dw.fill_zero();
    cw.fill_zero();
    bcw.fill_zero();
    abcw.fill_zero();

    //Do cw = Cw, then, bcw = B * (Cw), then abcw = A * (B * (Cw))
    unsafe {
        ref_gemm.run(c, &mut w, &mut cw, &ThreadInfo::single_thread() );
        ref_gemm.run(b, &mut cw, &mut bcw, &ThreadInfo::single_thread() );
        ref_gemm.run(a, &mut bcw, &mut abcw, &ThreadInfo::single_thread() );
    }

    //Do dw = Dw
    unsafe {
        ref_gemm.run(d, &mut w, &mut dw, &ThreadInfo::single_thread() );
    }

    //Dw -= abcw
    dw.axpy(T::zero() - T::one(), &abcw);
    dw.frosqr()
}

pub fn dur_seconds(start: Instant) -> f64 {
    let dur = start.elapsed();
    let time_secs = dur.as_secs() as f64;
    let time_nanos = dur.subsec_nanos() as f64;
    time_nanos / 1E9 + time_secs
}

pub fn gflops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
    let nflops = (m * n * k) as f64;
    2.0 * nflops / seconds / 1E9
}

pub fn gflops_ab(m: usize, n: usize, k: usize, l: usize, seconds: f64) -> f64 {
    let nflops = (m * k * l + m * l * n) as f64;
    2.0 * nflops / seconds / 1E9
}

pub fn gflops_bc(m: usize, n: usize, k: usize, l: usize, seconds: f64) -> f64 {
    let nflops = (k * l * n + m * k * n) as f64;
    2.0 * nflops / seconds / 1E9
}

pub fn gflops3(m: usize, n: usize, k: usize, l: usize, seconds: f64) -> f64 {
    let nflops = cmp::min(m * k * l + m * l * n, k * l * n + m * k * n) as f64;
    2.0 * nflops / seconds / 1E9
}

pub fn flush_cache(arr: &mut Vec<f64> ) {
    for i in (arr).iter_mut() {
        *i += 1.0;
    }
}

pub fn pin_to_core(core: usize) {
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

pub fn capacity_to_aligned_layout<T>(capacity: usize) -> Layout {
    Layout::new::<T>().repeat_packed(capacity).unwrap().align_to(4096)
}
