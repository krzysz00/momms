extern crate momms;
extern crate core;
extern crate typenum;
extern crate hwloc;
extern crate libc;

use std::time::{Instant};

pub use momms::matrix::{Scalar, Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix,
                        Subcomputation, TransposingSubcomputation};
pub use momms::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB,
                             ForceB, ForceA, TransposingOutputPanel, TripleLoop};
pub use momms::kern::{Ukernel, KernelNM, KernelMN};
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
    use typenum::{UInt, B0, Unsigned};
    type U3000 = UInt<UInt<typenum::U750, B0>, B0>;
    type Nc = U3000;
    type Kc = typenum::U192;
    type Mc = typenum::U120;
    type Mr = typenum::U4;
    type Nr = typenum::U12;

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

    type L3CiSub<T> = Subcomputation<T, Matrix<T>, Matrix<T>, ColPM<T>>;

    type L3Bo<T, MTA, MTAi, MTBi, MTC>
        = PartN<T, MTA, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC, L3CNc,
          PartK<T, MTA, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC, L3CNc, // Also the Mc of that algorithm, and outer k -> inner m
          ForceB<T, MTA, MTAi, MTBi, ColPM<T>, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC,
                    L3Ci<T, MTAi, MTBi, ColPM<T>>,
          PartM<T, MTA, ColPM<T>, MTC, McL2,
          PartK<T, MTA, ColPM<T>, MTC, Kc,
          PackA<T, MTA, ColPM<T>, MTC, RowPM<T>,
          KernelNM<T, RowPM<T>, ColPM<T>, MTC, Nr, Mr>>>>>>>;


    //Resident A algorithm
    type ColPMa<T> = ColumnPanelMatrix<T, Mr>;
    type RowPMa<T> = RowPanelMatrix<T, Nr>;
    type SubcompA<T, MTBTi, MTATi> = TransposingSubcomputation<T, MTBTi, MTATi, RowPMa<T>>;
    type NcL2 = typenum::U120;
    type L3Ao<T, MTBTi, MTATi, MTB, MTC>
        = PartM<T, SubcompA<T, MTBTi, MTATi>, MTB, MTC, L3CNc,
          PartK<T, SubcompA<T, MTBTi, MTATi>, MTB, MTC, L3CNc,
          ForceA<T, MTBTi, MTATi, RowPMa<T>, SubcompA<T, MTBTi, MTATi>, MTB, MTC,
                 TransposingOutputPanel<T, MTBTi, MTATi, RowPMa<T>, ColPM<T>,
                                        L3Ci<T, MTBTi, MTATi, ColPM<T>>>,
          PartN<T, RowPMa<T>, MTB, MTC, NcL2,
          PartK<T, RowPMa<T>, MTB, MTC, Kc,
          PackB<T, RowPMa<T>, MTB, MTC, ColPMa<T>,
          KernelMN<T, RowPMa<T>, ColPMa<T>, MTC, Nr, Mr>>>>>>>;
    let mut known = <L3Bo<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>>>::new();
    let mut ab = <L3Ao<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>>>::new();

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
        let (m, n, k, l) = (size, size, size, size);

        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a: Matrix<f64> = Matrix::new(m, k);
            let mut b: Matrix<f64> = Matrix::new(k, l);
            let mut c: Matrix<f64> = Matrix::new(l, n);
            let mut d: Matrix<f64> = Matrix::new_row_major(m, n);
            a.fill_rand(); b.fill_rand(); c.fill_rand(); d.fill_zero();
            let mut tmp: RowPMa<f64> = RowPanelMatrix::new(L3CNc::to_usize(), L3CNc::to_usize());
            tmp.fill_zero();
            b.transpose(); a.transpose();
            let mut submat: TransposingSubcomputation<f64, Matrix<f64>, Matrix<f64>, RowPMa<f64>>
                = TransposingSubcomputation::new(b, a, tmp);
            submat.set_scalar(0.0);
            flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                ab.run(&mut submat, &mut c, &mut d, &ThreadInfo::single_thread());
            }
            best_time = best_time.min(util::dur_seconds(start));
            submat.0.a.transpose(); submat.0.b.transpose();

            let err = test_d_eq_a_b_c(&mut submat.0.b, &mut submat.0.a, &mut c, &mut d);
            worst_err = worst_err.max(err);

            let mut tmp: ColPM<f64> = ColumnPanelMatrix::new(L3CNc::to_usize(), L3CNc::to_usize());
            tmp.fill_zero();
            let mut a_lifted = submat.0.b;
            let mut submat2: L3CiSub<f64> = Subcomputation::new(submat.0.a, c, tmp);
            //let mut submat: L3CiSub<f64> = submat.inner_transpose().set_c(tmp);
            submat2.set_scalar(0.0);
            flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                known.run(&mut a_lifted, &mut submat2, &mut d, &ThreadInfo::single_thread());
            }
            best_time_stock = best_time_stock.min(util::dur_seconds(start));
            ::std::mem::drop(submat2);
            ::std::mem::drop(a_lifted);
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{:e}",
                 m, n, k, l,
                 gflops_ab(m,n,k,l,best_time),
                 gflops_bc(m,n,k,l,best_time_stock),
                 worst_err.sqrt());
    }
    let sum: f64 = flusher.iter().sum();
    println!("# Flush value {}", sum);
}
fn main() {
    test_gemm3();
}
