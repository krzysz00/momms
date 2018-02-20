extern crate momms;
extern crate core;
extern crate typenum;
extern crate hwloc;
extern crate libc;

use typenum::{Unsigned};
use std::time::{Instant};

pub use momms::matrix::{ColumnPanelMatrix, Matrix, Mat};
pub use momms::composables::{GemmNode};
pub use momms::algorithms::{Dgemm3, GotoDgemm3,
                            Dgemm3Sub, GotoDgemm3Sub,
                            ColPM, Dgemm3TmpWidth, Dgemm3TmpHeight};
pub use momms::util;
pub use momms::thread_comm::ThreadInfo;

fn test_gemm3() {
    let mut chained = Dgemm3::new();
    let mut goto = GotoDgemm3::new();

    let flusher_len = 2*1024*1024; //16MB
    let mut flusher: Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len {
        flusher.push(0.0);
    }

    for index in 1..308 {//512 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_stock: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let size = index * 16;//8;
        let (m, n, k, l) = (size, size, size, size);

        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a: Matrix<f64> = Matrix::new(m, k);
            let mut b: Matrix<f64> = Matrix::new(k, l);
            let mut c: Matrix<f64> = Matrix::new(l, n);
            let mut d: Matrix<f64> = Matrix::new_row_major(m, n);
            a.fill_rand(); b.fill_rand(); c.fill_rand(); d.fill_zero();
            let mut tmp = ColPM::<f64>::new(Dgemm3TmpWidth::to_usize(),
                                            Dgemm3TmpHeight::to_usize());
            tmp.fill_zero();
            let mut submat = Dgemm3Sub::new(b, c, tmp);
            util::flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                chained.run(&mut a, &mut submat, &mut d, &ThreadInfo::single_thread());
            }
            best_time = best_time.min(util::dur_seconds(start));
            let err = util::test_d_eq_a_b_c(&mut a, &mut submat.a, &mut submat.b, &mut d);
            worst_err = worst_err.max(err);

            let tmp: Matrix<f64> = Matrix::new_row_major(k, n);
            let mut submat: GotoDgemm3Sub = submat.set_c(tmp); // drops old tmp
            submat.set_scalar(0.0);
            util::flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                goto.run(&mut a, &mut submat, &mut d, &ThreadInfo::single_thread());
            }
            best_time_stock = best_time_stock.min(util::dur_seconds(start));
            ::std::mem::drop(submat);
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{:e}",
                 m, n, k, l,
                 util::gflops_bc(m,n,k,l,best_time),
                 util::gflops_bc(m,n,k,l,best_time_stock),
                 worst_err.sqrt());
    }
    let sum: f64 = flusher.iter().sum();
    println!("# Flush value {}", sum);
}
fn main() {
    test_gemm3();
}
