extern crate momms;
extern crate core;
extern crate typenum;
extern crate hwloc;
extern crate libc;

use std::time::{Instant};

pub use momms::matrix::{Scalar, Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix,
                        Subcomputation};
pub use momms::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB,
                             ForceB, TripleLoop};
pub use momms::kern::{Ukernel, KernelNM};
pub use momms::util;
pub use momms::thread_comm::ThreadInfo;

fn test_gemm() {
    use typenum::{UInt, B0};
    type U3000 = UInt<UInt<typenum::U750, B0>, B0>;
    type Nc = U3000;
    type Kc = typenum::U192;
    type Mc = typenum::U120;
    type Mr = typenum::U4;
    type Nr = typenum::U12;

    type Goto<T, MTA, MTB, MTC> =
          PartN<T, MTA, MTB, MTC, Nc,
          PartK<T, MTA, MTB, MTC, Kc,
          PackB<T, MTA, MTB, MTC, ColumnPanelMatrix<T, Nr>,
          PartM<T, MTA, ColumnPanelMatrix<T,Nr>, MTC, Mc,
          PackA<T, MTA, ColumnPanelMatrix<T,Nr>, MTC, RowPanelMatrix<T,Mr>,
          KernelNM<T, RowPanelMatrix<T,Mr>, ColumnPanelMatrix<T,Nr>, MTC, Nr, Mr>>>>>>;

    type GotoPrime = Goto<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>;
    let mut goto: GotoPrime = GotoPrime::new();

    let flusher_len = 32*1024*1024; //256MB
    let mut flusher: Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len {
        flusher.push(0.0);
    }

    for index in 1..96 {//513 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_stock: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let size = index * 16;//8;
        let (m, n, k) = (size, size, size);

        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a: Matrix<f64> = Matrix::new(m, k);
            let mut b: Matrix<f64> = Matrix::new(k, n);
            let mut c: Matrix<f64> = Matrix::new_row_major(m, n);
            a.fill_rand(); b.fill_rand(); c.fill_zero();

            util::flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                goto.run(&mut a, &mut b, &mut c, &ThreadInfo::single_thread());
            }
            best_time = best_time.min(util::dur_seconds(start));
            let err = util::test_c_eq_a_b(&mut a, &mut b, &mut c);
            worst_err = worst_err.max(err);

            util::flush_cache(&mut flusher);

            let start = Instant::now();
            util::blis_dgemm(&mut a, &mut b, &mut c);
            best_time_stock = best_time_stock.min(util::dur_seconds(start));
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{:e}",
                 m, n, k, 0,
                 util::gflops(m,n,k,best_time),
                 util::gflops(m,n,k,best_time_stock),
                 worst_err.sqrt());
    }
    let sum: f64 = flusher.iter().sum();
    println!("# Flush value {}", sum);
}
fn main() {
    test_gemm();
}
