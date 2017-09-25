extern crate momms;
extern crate core;
extern crate typenum;
extern crate hwloc;
extern crate libc;

use std::time::{Instant};

pub use momms::matrix::{Scalar, Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix,
                        Subcomputation};
pub use momms::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB,
                             ForceB, ForceA, TripleLoop};
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
    type GotoChained = ForceB<f64,
                              Matrix<f64>,
                              Matrix<f64>, Matrix<f64>, Matrix<f64>, GotoChainedSub,
                              Matrix<f64>, GotoPrime, GotoPrime>;


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

    let mut chained = <L3Bo<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>>>::new();
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
        let (m, n, k, l) = (size, size, size, size);

        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a: Matrix<f64> = Matrix::new(m, k);
            let mut b: Matrix<f64> = Matrix::new(k, l);
            let mut c: Matrix<f64> = Matrix::new(l, n);
            let mut d: Matrix<f64> = Matrix::new(m, n);
            a.fill_rand(); b.fill_rand(); c.fill_rand(); d.fill_zero();
            let mut tmp: ColPM<f64> = ColumnPanelMatrix::new(L3CNc::to_usize(), L3CNc::to_usize());
            tmp.fill_zero();
            // All the transposing, since D += (AB)C \equiv D^T += C^T(B^T A^T)
            c.transpose(); b.transpose(); a.transpose(); d.transpose();
            let mut submat: L3CiSub<f64> = Subcomputation::new(b, a, tmp);
            flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                chained.run(&mut c, &mut submat, &mut d, &ThreadInfo::single_thread());
            }
            best_time = best_time.min(util::dur_seconds(start));
            // Comments are from a slightly less fair comparison
            //let mut b = submat.a;
            //let mut a = submat.b;
            //d.transpose(); c.transpose(); b.transpose(); a.transpose();
            // Here we recreate all the transposage
            let err = test_d_eq_a_b_c(&mut c, &mut submat.a, &mut submat.b, &mut d);
            worst_err = worst_err.max(err);

            let mut tmp: Matrix<f64> = Matrix::new_row_major(m, l);
            tmp.fill_zero();
            let mut submat: GotoChainedSub = submat.set_c(tmp); //Subcomputation::new(a, b, tmp);
            flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                goto.run(&mut c, &mut submat, &mut d, &&ThreadInfo::single_thread());
                //goto.run(&mut submat, &mut c, &mut d, &ThreadInfo::single_thread());
            }
            best_time_stock = best_time_stock.min(util::dur_seconds(start));
            ::std::mem::drop(submat);
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{:e}",
                 m, n, k, l,
                 gflops_ab(m,n,k,l,best_time),
                 gflops_ab(m,n,k,l,best_time_stock),
                 worst_err.sqrt());
    }
    let sum: f64 = flusher.iter().sum();
    println!("# Flush value {}", sum);
}
fn main() {
    test_gemm3();
}
