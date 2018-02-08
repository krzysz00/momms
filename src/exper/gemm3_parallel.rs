extern crate momms;
extern crate typenum;
extern crate hwloc;
extern crate libc;

use std::time::{Instant};

pub use momms::matrix::{Scalar, Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix,
                        Subcomputation};
pub use momms::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB,
                             ForceB, TripleLoop, SpawnThreads, ParallelM, ParallelN, TheRest};
pub use momms::kern::{Ukernel, KernelNM};
pub use momms::util;
pub use momms::thread_comm::ThreadInfo;

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
          ParallelN<T, RowPanelMatrix<T,Mr>, ColumnPanelMatrix<T,Nr>, MTC, Nr, TheRest,
          KernelNM<T, RowPanelMatrix<T,Mr>, ColumnPanelMatrix<T,Nr>, MTC, Nr, Mr>>>>>>>;

    type GotoPrime = Goto<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>;
    type GotoChainedSub = Subcomputation<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>;
    type GotoChained = SpawnThreads<f64, Matrix<f64>, GotoChainedSub, Matrix<f64>,
                              ForceB<f64, Matrix<f64>,
                              Matrix<f64>, Matrix<f64>, Matrix<f64>, GotoChainedSub,
                              Matrix<f64>, GotoPrime, GotoPrime>>;


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
          ParallelN<T, RowPM<T>, ColPM<T>, MTC, Nr, TheRest,
          KernelNM<T, RowPM<T>, ColPM<T>, MTC, Nr, Mr>>>>>>;

    type L3CiSub<T> = Subcomputation<T, Matrix<T>, Matrix<T>, ColPM<T>>;

    type L3Bo<T, MTA, MTAi, MTBi, MTC> =
          SpawnThreads<T, MTA, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC,
          PartN<T, MTA, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC, L3CNc,
          PartK<T, MTA, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC, L3CNc, // Also the Mc of that algorithm, and outer k -> inner m
          ForceB<T, MTA, MTAi, MTBi, ColPM<T>, Subcomputation<T, MTAi, MTBi, ColPM<T>>, MTC,
                    L3Ci<T, MTAi, MTBi, ColPM<T>>,
          PartM<T, MTA, ColPM<T>, MTC, McL2,
          PartK<T, MTA, ColPM<T>, MTC, Kc,
          PackA<T, MTA, ColPM<T>, MTC, RowPM<T>,
          ParallelN<T, RowPM<T>, ColPM<T>, MTC, Nr, TheRest,
          KernelNM<T, RowPM<T>, ColPM<T>, MTC, Nr, Mr>>>>>>>>>;

    let mut chained = <L3Bo<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>>>::new();
    let mut goto: GotoChained = GotoChained::new();

    let topology = ::hwloc::Topology::new();
    let n_cores = topology.objects_with_type(&::hwloc::ObjectType::Core).unwrap().len();
    let n_cores_float: f64 = (n_cores as u32).into();
    chained.set_n_threads(n_cores);
    goto.set_n_threads(n_cores);
    println!{"# Number of cores: {}", n_cores};

    let flusher_len = 32*1024*1024; //256MB
    let mut flusher: Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len {
        flusher.push(0.0);
    }

    for index in 1..513 {//96 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_stock: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let size = index * 8;//16;
        let (m, n, k, l) = (size, size, size, size);

        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a: Matrix<f64> = Matrix::new(m, k);
            let mut b: Matrix<f64> = Matrix::new(k, l);
            let mut c: Matrix<f64> = Matrix::new(l, n);
            let mut d: Matrix<f64> = Matrix::new_row_major(m, n);
            a.fill_rand(); b.fill_rand(); c.fill_rand(); d.fill_zero();
            let mut tmp: ColPM<f64> = ColumnPanelMatrix::new(L3CNc::to_usize(), L3CNc::to_usize());
            tmp.fill_zero();
            let mut submat: L3CiSub<f64> = Subcomputation::new(b, c, tmp);
            util::flush_cache(&mut flusher);

            let start = Instant::now();
            unsafe {
                chained.run(&mut a, &mut submat, &mut d, &ThreadInfo::single_thread());
            }
            best_time = best_time.min(util::dur_seconds(start));
            let err = util::test_d_eq_a_b_c(&mut a, &mut submat.a, &mut submat.b, &mut d);
            worst_err = worst_err.max(err);

            let tmp: Matrix<f64> = Matrix::new_row_major(k, n);
            let mut submat: GotoChainedSub = submat.set_c(tmp); // drops old tmp
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
                 util::gflops_bc(m,n,k,l,best_time) / n_cores_float,
                 util::gflops_bc(m,n,k,l,best_time_stock) / n_cores_float,
                 worst_err.sqrt());
    }
    let sum: f64 = flusher.iter().sum();
    println!("# Flush value {}", sum);
}
fn main() {
    test_gemm3();
}
