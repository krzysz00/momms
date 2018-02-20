extern crate typenum;
use matrix::{ColumnPanelMatrix, RowPanelMatrix, Matrix, Subcomputation};
use composables::{PartM, PartN, PartK, PackA, PackB,
                  ForceB, SpawnThreads, ParallelN, TheRest};
use kern::{KernelNM};
use typenum::{UInt, B0};

// BLIS's constants
type U4080 = UInt<UInt<typenum::U1020, B0>, B0>;
type Nc = U4080;
type Kc = typenum::U256;
type Mc = typenum::U72;
type Mr = typenum::U6;
type Nr = typenum::U8;

// Old constants
// type U3000 = UInt<UInt<typenum::U750, B0>, B0>;
// type Nc = U3000;
// type Kc = typenum::U192;
// type Mc = typenum::U120;
// type Mr = typenum::U4;
// type Nr = typenum::U12;

type Goto<T, MTA, MTB, MTC> =
      PartN<T, MTA, MTB, MTC, Nc,
      PartK<T, MTA, MTB, MTC, Kc,
      PackB<T, MTA, MTB, MTC, ColumnPanelMatrix<T, Nr>,
      PartM<T, MTA, ColumnPanelMatrix<T,Nr>, MTC, Mc,
      PackA<T, MTA, ColumnPanelMatrix<T,Nr>, MTC, RowPanelMatrix<T,Mr>,
      ParallelN<T, RowPanelMatrix<T,Mr>, ColumnPanelMatrix<T,Nr>, MTC, Nr, TheRest,
      KernelNM<T, RowPanelMatrix<T,Mr>, ColumnPanelMatrix<T,Nr>, MTC, Nr, Mr>>>>>>>;

pub type GotoDgemm = Goto<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>;
pub type GotoDgemm3Sub = Subcomputation<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>;
pub type GotoDgemm3 =
    SpawnThreads<f64, Matrix<f64>, GotoDgemm3Sub, Matrix<f64>,
    ForceB<f64, Matrix<f64>,
           Matrix<f64>, Matrix<f64>, Matrix<f64>, GotoDgemm3Sub,
           Matrix<f64>,
    GotoDgemm, GotoDgemm>>;

    // type RootS3 = typenum::U768;
type McL2 = typenum::U72; //typenum::U120;
pub type ColPM<T> = ColumnPanelMatrix<T, Nr>;
pub type RowPM<T> = RowPanelMatrix<T, Mr>;

type U2040 = UInt<typenum::U1020, B0>;
type L3CNc = U2040;
type L3CMc = typenum::U252;
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
pub type Dgemm3Sub = L3CInnerSub<f64>;
pub type Dgemm3TmpWidth = L3CNc;
pub type Dgemm3TmpHeight = L3CNc;
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

pub type Dgemm3 = L3BOuter<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>>;
