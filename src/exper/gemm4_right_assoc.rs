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

fn test_e_eq_a_b_c_d<T:Scalar, At: Mat<T>, Bt: Mat<T>,
                     Ct: Mat<T>, Dt: Mat<T>, Et: Mat<T>>(a: &mut At, b: &mut Bt,
                                                         c: &mut Ct, d: &mut Dt, e: &mut Et) -> T {
    let mut ref_gemm : TripleLoop = TripleLoop{};

    let d0 = e.height();
    let d4 = d.width();
    let d3 = c.width();
    let d2 = b.width();
    let d1 = a.width();

    let mut w: Matrix<T> = Matrix::new(d4, 1);
    let mut dw: Matrix<T> = Matrix::new(d3, 1);
    let mut cdw: Matrix<T> = Matrix::new(d2, 1);
    let mut bcdw: Matrix<T> = Matrix::new(d1, 1);
    let mut abcdw: Matrix<T> = Matrix::new(d0, 1);
    let mut ew: Matrix<T> = Matrix::new(d1, 1);
    w.fill_rand();
    ew.fill_zero();
    dw.fill_zero();
    cdw.fill_zero();
    bcdw.fill_zero();
    abcdw.fill_zero();

    unsafe {
        ref_gemm.run(d, &mut w, &mut dw, &ThreadInfo::single_thread() );
        ref_gemm.run(c, &mut dw, &mut cdw, &ThreadInfo::single_thread() );
        ref_gemm.run(b, &mut cdw, &mut bcdw, &ThreadInfo::single_thread() );
        ref_gemm.run(a, &mut bcdw, &mut abcdw, &ThreadInfo::single_thread() );
    }

    //Do ew = Ew
    unsafe {
        ref_gemm.run(e, &mut w, &mut ew, &ThreadInfo::single_thread() );
    }

    //Dw -= abcw
    ew.axpy(T::zero() - T::one(), &abcdw);
    ew.frosqr()
}

fn gflops_right(d0: usize, d1: usize, d2: usize, d3: usize, d4: usize, seconds: f64) -> f64 {
    let nflops = (d2 * d3 * d4 + d1 * d2 * d4 + d0 * d1 * d4) as f64;
    2.0 * nflops / seconds / 1E9
}

fn flush_cache(arr: &mut Vec<f64> ) {
    for i in (arr).iter_mut() {
        *i += 1.0;
    }
}

fn test_gemm4() {
    use typenum::{UInt, B0, Unsigned};
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
    type GotoChainedSub = Subcomputation<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>;
    type GotoChainedSubSub = Subcomputation<f64, Matrix<f64>, GotoChainedSub, Matrix<f64>>;

    type GotoChained = ForceB<f64, Matrix<f64>, Matrix<f64>, GotoChainedSub, Matrix<f64>,
                              GotoChainedSubSub, Matrix<f64>,
                                  ForceB<f64, Matrix<f64>,
                                  Matrix<f64>, Matrix<f64>, Matrix<f64>, GotoChainedSub,
                                  Matrix<f64>, GotoPrime, GotoPrime>,
                              GotoPrime>;


    // type RootS3 = typenum::U768;
    type McL2 = typenum::U120;
    type ColPM<T> = ColumnPanelMatrix<T, Nr>;
    type RowPM<T> = RowPanelMatrix<T, Mr>;

    type L3CNc = typenum::U624;
    type L3CKc = typenum::U156;
    //Resident C algorithm, inner loops
    type L3Ci<T, MTA, MTB, MTC> =
          PartK<T, MTA, MTB, MTC, L3CKc,
          PackB<T, MTA, MTB, MTC, ColPM<T>,
          PartM<T, MTA, ColPM<T>, MTC, L3CKc,
          PackA<T, MTA, ColPM<T>, MTC, RowPM<T>,
          KernelNM<T, RowPM<T>, ColPM<T>, MTC, Nr, Mr>>>>>;
    type L3CiSub<T, MTA, MTB> = Subcomputation<T, MTA, MTB, ColPM<T>>;

    type L3Cm<T, MTA, MTAi, MTBi, MTC> =
          PartK<T, MTA, L3CiSub<T, MTAi, MTBi>, MTC, L3CKc,
          ForceB<T, MTA, MTAi, MTBi, ColPM<T>, L3CiSub<T, MTAi, MTBi>, MTC,
                 L3Ci<T, MTAi, MTBi, ColPM<T>>,
          PartM<T, MTA, ColPM<T>, MTC, L3CKc,
          PackA<T, MTA, ColPM<T>, MTC, RowPM<T>,
          KernelNM<T, RowPM<T>, ColPM<T>, MTC, Nr, Mr>>>>>;

    type L3CmSub<T, MTB, MTC, MTD> = Subcomputation<T, MTB, L3CiSub<T, MTC, MTD>, ColPM<T>>;

    type L3Bo<T, MTA, MTB, MTC, MTD, MTE>
        = PartN<T, MTA, L3CmSub<T, MTB, MTC, MTD>, MTE, L3CNc,
          PartK<T, MTA, L3CmSub<T, MTB, MTC, MTD>, MTE, L3CNc, // Also the Mc of that algorithm, and outer k -> inner m
          ForceB<T, MTA, MTB, L3CiSub<T, MTC, MTD>, ColPM<T>, L3CmSub<T, MTB, MTC, MTD>, MTE,
                    L3Cm<T, MTB, MTC, MTD, ColPM<T>>,
          PartM<T, MTA, ColPM<T>, MTE, McL2,
          PartK<T, MTA, ColPM<T>, MTE, Kc,
          PackA<T, MTA, ColPM<T>, MTE, RowPM<T>,
          KernelNM<T, RowPM<T>, ColPM<T>, MTE, Nr, Mr>>>>>>>;

    let mut chained = <L3Bo<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>>>::new();
    let mut goto: GotoChained = GotoChained::new();

    let flusher_len = 32*1024*1024; //256MB
    let mut flusher: Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len {
        flusher.push(0.0);
    }

    for index in 1..96 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_stock: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let size = index * 16;
        let (d0, d1, d2, d3, d4) = (size, size, size, size, size);

        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a: Matrix<f64> = Matrix::new(d0, d1);
            let mut b: Matrix<f64> = Matrix::new(d1, d2);
            let mut c: Matrix<f64> = Matrix::new(d2, d3);
            let mut d: Matrix<f64> = Matrix::new(d3, d4);
            let mut e: Matrix<f64> = Matrix::new_row_major(d0, d4);

            a.fill_rand(); b.fill_rand(); c.fill_rand(); d.fill_rand(); e.fill_zero();
            let mut tmp_m: ColPM<f64> = ColumnPanelMatrix::new(L3CNc::to_usize(), L3CNc::to_usize());
            let mut tmp_i: ColPM<f64> = ColumnPanelMatrix::new(L3CNc::to_usize(), L3CNc::to_usize());
            tmp_m.fill_zero();
            tmp_i.fill_zero();
            let mut submat_i: L3CiSub<f64, Matrix<f64>, Matrix<f64>> = Subcomputation::new(c, d, tmp_i);
            let mut submat_m: L3CmSub<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>
                = Subcomputation::new(b, submat_i, tmp_m);
            flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                chained.run(&mut a, &mut submat_m, &mut e, &ThreadInfo::single_thread());
            }
            best_time = best_time.min(util::dur_seconds(start));
            let err = test_e_eq_a_b_c_d(&mut a, &mut submat_m.a,
                                        &mut submat_m.b.a, &mut submat_m.b.b, &mut e);
            worst_err = worst_err.max(err);

            let mut tmp_m: Matrix<f64> = Matrix::new_row_major(d2, d4);
            let mut tmp_i: Matrix<f64> = Matrix::new_row_major(d3, d4);
            tmp_i.fill_zero();
            tmp_m.fill_zero();
            let mut b_prime = submat_m.a;
            let mut c_prime = submat_m.b.a;
            let mut d_prime = submat_m.b.b;
            let mut submat_i = Subcomputation::new(c_prime, d_prime, tmp_i);
            let mut submat_m = Subcomputation::new(b_prime, submat_i, tmp_m);
//            let mut submat_m = submat_m.set_b(submat_m.b.set_c(tmp_i)).set_c(tmp_m);
            flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                goto.run(&mut a, &mut submat_m, &mut e, &ThreadInfo::single_thread());
            }
            best_time_stock = best_time_stock.min(util::dur_seconds(start));
            ::std::mem::drop(submat_m);
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:e}",
                 d0, d1, d2, d3, d4,
                 gflops_right(d0,d1,d2,d3,d4,best_time),
                 gflops_right(d0,d1,d2,d3,d4,best_time_stock),
                 worst_err.sqrt());
    }
    let sum: f64 = flusher.iter().sum();
    println!("# Flush value {}", sum);
}
fn main() {
    test_gemm4();
}
