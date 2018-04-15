extern crate momms;
extern crate core;
extern crate typenum;
extern crate hwloc;
extern crate libc;

use std::time::{Instant};

pub use momms::matrix::{Matrix, Mat};
pub use momms::composables::{GemmNode};
pub use momms::algorithms::{GotoDgemm};
pub use momms::util;
pub use momms::thread_comm::ThreadInfo;

fn test_gemm() {
    let mut goto = GotoDgemm::new();

    let flusher_len = 4*1024*1024; //32MB
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
