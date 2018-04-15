extern crate momms;
extern crate core;
extern crate typenum;
extern crate hwloc;
extern crate libc;

use typenum::{Unsigned};
use std::time::{Instant};

pub use momms::matrix::{ColumnPanelMatrix, Matrix, Mat};
pub use momms::composables::{GemmNode};
pub use momms::algorithms::{Dgemm3,
                            Dgemm3Sub,
                            ColPM, Dgemm3TmpWidth, Dgemm3TmpHeight};
pub use momms::util;
pub use momms::thread_comm::ThreadInfo;

pub use momms::matrix::{RowPanelMatrix, Subcomputation};
pub use momms::composables::{PartM, PartN, PartK, PackA, PackB,
                             ForceB, SpawnThreads, ParallelN, TheRest};
pub use momms::kern::{KernelNM};
use typenum::{UInt, B0};

// BLIS's constants
//type Mc = typenum::U72;
type Mr = typenum::U6;
type Nr = typenum::U8;

type McL2 = typenum::U72; //typenum::U120;
pub type RowPM<T> = RowPanelMatrix<T, Mr>;

type U2040 = UInt<typenum::U1020, B0>;
type L3CNc = U2040;
type L3CMc = typenum::U256;
type L3CKc = typenum::U256;
//Resident C algorithm, inner loops
type L3CInner<T, MTA, MTB, MTC> =
      PartK<T, MTA, MTB, MTC, L3CKc,
      PackB<T, MTA, MTB, MTC, ColPM<T>,
      PartM<T, MTA, ColPM<T>, MTC, McL2,
      PackA<T, MTA, ColPM<T>, MTC, RowPM<T>,
      ParallelN<T, RowPM<T>, ColPM<T>, MTC, Nr, TheRest,
      KernelNM<T, RowPM<T>, ColPM<T>, MTC, Nr, Mr>>>>>>;

type L3CInnerSub<T> = Subcomputation<T, Matrix<T>, Matrix<T>, ColPM<T>>;
pub type Dgemm3BSub = L3CInnerSub<f64>;
pub type Dgemm3BTmpWidth = L3CNc;
pub type Dgemm3BTmpHeight = L3CNc;
type L3BOuter<T, MTA, MTAi, MTBi, MTC> =
      SpawnThreads<T, MTA, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC,
      PartN<T, MTA, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC, L3CNc,
      PartK<T, MTA, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC, L3CMc, // Also the Mc of that algorithm, and outer k -> inner m
      ForceB<T, MTA, MTAi, MTBi, ColPM<T>, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC,
                L3CInner<T, MTAi, MTBi, ColPM<T>>,
      PartM<T, MTA, ColPM<T>, MTC, McL2,
//    PartK<T, MTA, ColPM<T>, MTC, Kc,
      PackA<T, MTA, ColPM<T>, MTC, RowPM<T>,
      ParallelN<T, RowPM<T>, ColPM<T>, MTC, Nr, TheRest,
      KernelNM<T, RowPM<T>, ColPM<T>, MTC, Nr, Mr>>>>>>>>;

pub type Dgemm3B = L3BOuter<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>>;

fn test_gemm3() {
    let mut chained = Dgemm3B::new();
    let mut goto = Dgemm3::new();

    let flusher_len = 4*1024*1024; //32MB
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
            let mut tmp = ColPM::<f64>::new(Dgemm3BTmpWidth::to_usize(),
                                            Dgemm3BTmpHeight::to_usize());
            tmp.fill_zero();
            let mut submat = Dgemm3BSub::new(b, c, tmp);
            util::flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                chained.run(&mut a, &mut submat, &mut d, &ThreadInfo::single_thread());
            }
            best_time = best_time.min(util::dur_seconds(start));
            let err = util::test_d_eq_a_b_c(&mut a, &mut submat.a, &mut submat.b, &mut d);
            worst_err = worst_err.max(err);

            let mut tmp = ColPM::<f64>::new(Dgemm3TmpWidth::to_usize(),
                                            Dgemm3TmpHeight::to_usize());
            tmp.fill_zero();
            let mut submat: Dgemm3Sub = submat.set_c(tmp); // drops old tmp
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
