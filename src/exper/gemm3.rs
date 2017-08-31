extern crate momms;
extern crate core;
extern crate typenum;
extern crate hwloc;
extern crate libc;

use std::time::{Instant};
use typenum::{U1};

pub use momms::matrix::{Scalar, Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix,
                        Subcomputation};
pub use momms::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB,
                             ForceB, TripleLoop};
pub use momms::kern::{Ukernel, KernelNM};
pub use momms::util;
pub use momms::thread_comm::ThreadInfo;
fn test_d_eq_a_b_c<T:Scalar, At: Mat<T>, Bt: Mat<T>,
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

fn gflops_ab(m: usize, n: usize, k: usize, l: usize, seconds: f64) -> f64 {
    let nflops = (m * k * l + m * l * n) as f64;
    2.0 * nflops / seconds / 1E9
}

fn gflops_bc(m: usize, n: usize, k: usize, l: usize, seconds: f64) -> f64 {
    let nflops = (k * l * n + m * k * n) as f64;
    2.0 * nflops / seconds / 1E9
}

fn flush_cache(arr: &mut Vec<f64> ) {
    for i in (arr).iter_mut() {
        *i += 1.0;
    }
}

fn test_gemm3() {
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

    type GotoChainedSub = Subcomputation<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>;
    type GotoChained = ForceB<f64, Matrix<f64>,
                              Matrix<f64>, Matrix<f64>, Matrix<f64>,
                              Matrix<f64>, GotoPlain, GotoPlain>;

    let mut chained: GotoChained = GotoChained::new();
    let mut stock: GotoPlain = GotoPlain::new();

    let flusher_len = 32*1024*1024; //256MB
    let mut flusher: Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len {
        flusher.push(0.0);
    }

    for index in 1..64 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_stock: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let size = index * 16;
        let (m, n, k, l) = (size, size, size, size);

        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a: Matrix<f64> = Matrix::new(m, k);
            let mut b: Matrix<f64> = Matrix::new(k, l);
            let mut c: Matrix<f64> = Matrix::new(l, n);
            let mut d: Matrix<f64> = Matrix::new_row_major(m, n);
            a.fill_rand(); b.fill_rand(); c.fill_rand(); d.fill_zero();
            let mut tmp: Matrix<f64> = Matrix::new_row_major(k, n);

            flush_cache(&mut flusher);

            let mut submat: GotoChainedSub = Subcomputation::new(b, c, tmp);
            let mut start = Instant::now();
            submat.set_scalar(0.0);
            unsafe {
                chained.run(&mut a, &mut submat, &mut d, &ThreadInfo::single_thread());
            }
            best_time = best_time.min(util::dur_seconds(start));
            let err = test_d_eq_a_b_c(&mut a, &mut submat.a, &mut submat.b, &mut d);
            worst_err = worst_err.max(err);

            flush_cache(&mut flusher);

            let mut tmp: Matrix<f64> = Matrix::new_row_major(k, n);
            tmp.set_scalar(0.0);
            let mut start = Instant::now();
            unsafe {
                stock.run(&mut submat.a, &mut submat.b, &mut tmp, &ThreadInfo::single_thread());
                stock.run(&mut a, &mut tmp, &mut d, &ThreadInfo::single_thread());
            }
            best_time_stock = best_time_stock.min(util::dur_seconds(start));
            ::std::mem::drop(tmp);
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{:e}",
                 m, n, k, l,
                 gflops_bc(m,n,k,l,best_time),
                 gflops_bc(m,n,k,l,best_time_stock),
                 worst_err.sqrt());
    }
    let sum: f64 = flusher.iter().sum();
    println!("Flush value {}", sum);
}
fn main() {
    test_gemm3();
}
