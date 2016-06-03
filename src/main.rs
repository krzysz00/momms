#![feature(specialization)]
#![feature(zero_one)]
extern crate core;
extern crate rand;

use std::ops::{Add, Mul, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign};
use std::num::{Zero};
use core::fmt::{Display};
use core::marker::{PhantomData, Sized};

trait Scalar where
    Self: Add<Self, Output=Self>,
    Self: Mul<Self, Output=Self>,
    Self: Sub<Self, Output=Self>,
    Self: Div<Self, Output=Self>,
    Self: AddAssign<Self>,
    Self: MulAssign<Self>,
    Self: SubAssign<Self>,
    Self: DivAssign<Self>,
    Self: Zero,
    Self: Sized,
    Self: Copy,
    Self: Display,
    Self: rand::Rand,
{}
impl Scalar for f64{}
impl Scalar for f32{}

trait Mat<T: Scalar> {
    fn get( &self, y: usize, x: usize) -> T;
    fn set( &mut self, y: usize, x: usize, alpha: T) -> ()
            where T: Sized;
    
    fn width( &self ) -> usize;
    fn height( &self ) -> usize;

    fn printWolfram( &self ) -> () {
        print!("{{");
        for y in 0..self.height() {
            print!("{{");
            for x in 0..self.width() {
                let formatted_number = format!("{:.*}", 2, self.get(y,x));
                print!("{}", formatted_number);
                if x < self.height() - 1 {
                    print!(",");
                }
            }
            print!("}}");
            if y < self.width() - 1 {
                print!(",");
            }
        }
        print!("}}");
    }

    fn print( &self ) -> () {
        for y in 0..self.width() {
            for x in 0..self.height() {
                let formatted_number = format!("{:.*}", 2, self.get(y,x));
                print!("{} ", formatted_number);
            }
        }
    }

    fn fill_rand( &mut self ) -> () {
        for x in 0..self.width() {
            for y in 0..self.height() {
                self.set(y,x,rand::random::<T>());
            }
        }
    }

    fn fill_zero( &mut self ) -> () {
        for x in 0..self.width() {
            for y in 0..self.height() {
                self.set(y,x,T::zero());
            }
        }
    }

    fn frosqr( &self ) -> T {
        let mut norm:T = T::zero();
        for x in 0..self.width() {
            for y in 0..self.height() {
                norm += self.get(y,x) * self.get(y,x);
            }
        }
        return norm;
    }
}

struct Matrix<T: Scalar> {
    h: usize,
    w: usize,
    row_stride: usize,
    column_stride: usize,
    buffer: Vec<T>,
}
impl<T: Scalar> Matrix<T> {
    fn new( h: usize, w: usize ) -> Matrix<T> {
        let mut buf = Vec::with_capacity( h * w );
        unsafe{ buf.set_len( h * w );}
        let mat = Matrix{ h: h, w: w, row_stride: 1, column_stride: h, buffer: buf };
        return mat;
    }
}
impl<T: Scalar> Mat<T> for Matrix<T> {
    fn get( &self, y: usize, x: usize) -> T {
        return self.buffer[y*self.row_stride + x*self.column_stride];
    }
    fn set( &mut self, y: usize, x: usize, alpha: T) -> () {
        self.buffer[y*self.row_stride + x*self.column_stride] = alpha;
    }
    fn width( &self ) -> usize {
        return self.w;
    }
    fn height( &self ) -> usize {
        return self.h;
    }
}

/*
struct ColumnPanelMatrix<T: Scalar> {
    h: usize,
    w: usize,
    panel_h: usize,
    panel_w: usize,
    panel_stride: usize,
    buffer: Vec<T>,
}
impl<T: Scalar> Mat<T> for ColumnPanelMatrix<T> {
    fn get( &self, x: usize, y:usize ) -> T {
        let panel_id = y / self.panel_w;
        let panel_index  = y % self.panel_w;
        let elem_index = panel_id * self.panel_stride + x * self.panel_w + panel_index;

        return self.buffer[ elem_index ];
    }
    fn set( &mut self, x: usize, y:usize, alpha: T) -> () {
        let panel_id = y / self.panel_w;
        let panel_index  = y % self.panel_w;
        let elem_index = panel_id * self.panel_stride + x * self.panel_w + panel_index;

        self.buffer[ elem_index ] = alpha;
    }
    fn height( &self ) -> usize{
        return self.h;
    }
    fn width( &self ) -> usize{
        return self.w;
    }
}*/

struct View {
    m: usize,
    n: usize,
    k: usize,
    m_off: usize,
    n_off: usize,
    k_off: usize,
}
impl View{
    fn new<T: Scalar>( a: &Mat<T>, b: &Mat<T>, c: &Mat<T> ) -> View{
        if a.height() != c.height() || a.width() != b.height() || c.width() != b.width() {
            panic!("Cannot create View with nonconformal operands!");
        }

        let v : View = View{ m: c.height(), n: b.width(), k: a.width(),
                      m_off: 0, n_off: 0, k_off: 0 };
        return v;
    }
}

trait GemmNode<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> {
     fn run( &self, a: &At, b: &Bt, c: &mut Ct, v: &mut View ) -> ();
}

/*
fn pack<T: Scalar>( a: &Matrix<T> ) {
}

struct PackA<T: Scalar, S: GemmNode<T>> {
    child: S,
    panel_height: usize,
    panel_width: usize,
    dt: PhantomData<T>,
}
impl<T:Scalar, S:GemmNode<T>> GemmNode<T> for PackA<T,S> {
     fn run( &self, a: &Matrix<T>, b: &Matrix<T>, c: &mut Matrix<T>, v: &mut View<T> ) {
        pack( a );
        self.child.run(a, b, c, v);
    }
}

struct PackB<T: Scalar, S: GemmNode<T>> {
    child: S,
    panel_height: usize,
    panel_width: usize,
    _dt: PhantomData<T>,
}
impl<T:Scalar, S:GemmNode<T>> GemmNode<T> for PackB<T,S> {
     fn run( &self, a: &Matrix<T>, b: &Matrix<T>, c: &mut Matrix<T>, v: &mut View<T> ) {
        pack( b );
        self.child.run(a, b, c, v);
    }
}
*/
struct PartM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    bsz: usize,
    iota: usize,
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> PartM<T,At,Bt,Ct,S> {
    fn new( bsz: usize, iota: usize, child: S ) -> PartM<T, At, Bt, Ct, S>{
        return PartM{ bsz: bsz, iota: iota, child: child, 
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }; 
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartM<T,At,Bt,Ct,S> {
    fn run( &self, a: &At, b: &Bt, c:&mut Ct, v: &mut View ) -> () {
        let m_save = v.m;
        let m_off_save = v.m_off;
        
        let mut i = 0;
        while i < m_save  {
            let bsz_step = std::cmp::min( self.bsz, m_save - i );   
            v.m = bsz_step;
            v.m_off = m_off_save + i;
            self.child.run(a, b, c, v);
            i += bsz_step;
        }
        v.m = m_save;
        v.m_off = m_off_save;
    }
}

struct PartN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    bsz: usize,
    iota: usize,
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> PartN<T,At,Bt,Ct,S> {
    fn new( bsz: usize, iota: usize, child: S ) -> PartN<T, At, Bt, Ct, S>{
        return PartN{ bsz: bsz, iota: iota, child: child, 
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }; 
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartN<T,At,Bt,Ct,S> {
    fn run( &self, a: &At, b: &Bt, c:&mut Ct, v: &mut View ) -> () {
        let n_save = v.n;
        let n_off_save = v.n_off;
        
        let mut i = 0;
        while i < n_save  {
            let bsz_step = std::cmp::min( self.bsz, n_save - i );   
            v.n = bsz_step;
            v.n_off = n_off_save + i;
            self.child.run(a, b, c, v);
            i += bsz_step;
        }
        v.n = n_save;
        v.n_off = n_off_save;
    }
}

struct PartK<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    bsz: usize,
    iota: usize,
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> PartK<T,At,Bt,Ct,S> {
    fn new( bsz: usize, iota: usize, child: S ) -> PartK<T, At, Bt, Ct, S>{
        return PartK{ bsz: bsz, iota: iota, child: child, 
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }; 
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartK<T,At,Bt,Ct,S> {
    fn run( &self, a: &At, b: &Bt, c:&mut Ct, v: &mut View ) -> () {
        let k_save = v.k;
        let k_off_save = v.k_off;
        
        let mut i = 0;
        while i < k_save  {
            let bsz_step = std::cmp::min( self.bsz, k_save - i );   
            v.k = bsz_step;
            v.k_off = k_off_save + i;
            self.child.run(a, b, c, v);
            i += bsz_step;
        }
        v.k = k_save;
        v.k_off = k_off_save;
    }
}

//Leaf node that does nothing. For testing.
/*struct End { }
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> GemmNode<T, At, Bt, Ct> for End {
    fn run( &self, a: &At, b: &Bt, c: &mut Ct, v: &mut View ) -> () {
    }
}*/

struct TripleLoopKernel{}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> 
    GemmNode<T, At, Bt, Ct> for TripleLoopKernel {
    fn run( &self, a: &At, b: &Bt, c: &mut Ct, v: &mut View ) -> () {
        //For now, let's do an axpy based gemm
        for jn in 0..v.n {
            for pk in 0..v.k {
                for im in 0..v.m {
                    let y = im+v.m_off;
                    let x = jn+v.n_off;
                    let z = pk+v.k_off;
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set( y, x, t );
                }
            }
        }
    }
}

fn axpy<T:Scalar, Xt: Mat<T>, Yt: Mat<T>>( alpha: T, x: &Xt, beta: T, y: &mut Yt ) {
    //x.h should equal y.h and x.w should equal y.w!
    //but we dont check this yet
    for i in 0..x.width() {
        for j in 0..x.height() {
            let t = alpha * x.get(j,i) + beta * y.get(j,i);
            y.set(j,i,t); 
        }
    }
}

fn main() {
    let m = 2;
    let n = 2;
    let k = 2;

    let mut a : Matrix<f64> = Matrix::new(m, k);
    let mut b : Matrix<f64> = Matrix::new(k, n);
    let mut c : Matrix<f64> = Matrix::new(m, n);
    let mut gemm_view : View = View::new(&a,&b,&c);
    a.fill_rand();
    b.fill_rand();
    c.fill_zero();


    let mut w : Matrix<f64> = Matrix::new(n, 1);
    let mut bw : Matrix<f64> = Matrix::new(k, 1);
    let mut abw : Matrix<f64> = Matrix::new(m, 1);
    let mut cw : Matrix<f64> = Matrix::new(m, 1);
    w.fill_rand();
    cw.fill_zero();
    bw.fill_zero();
    abw.fill_zero();


    //You can be explicit with the types like here:
    //The problem is that the types grow quadratically with the number of steps in the control tree!
    let kernel : TripleLoopKernel = TripleLoopKernel{};
    let loop1 : PartM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel> = PartM::new(5, 1, kernel);
    let loop2 : PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel>> = PartK::new(5, 1, loop1);
    let loop3 : PartN<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel>>> = PartN::new(5, 1, loop2);

    //If you instead delcare the phantom data here, the type inference system works
    //I.E. the compiler can infer the type of loop1_inf when you use that type in loop2_inf
    //Now the code grows linearly with the number of steps in our control tree
    //The problem with this is that we have all this ugly phantom data laying around!
    let kernel_inf = TripleLoopKernel{};
    let loop1_inf = PartM{ bsz: 5, iota: 1, child: kernel_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop2_inf = PartK{ bsz: 5, iota: 1, child: loop1_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop3_inf = PartN{ bsz: 5, iota: 1, child: loop2_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };

    let ref_gemm : TripleLoopKernel = TripleLoopKernel{};
    //C = AB
    loop3.run( &a, &b, &mut c, &mut gemm_view );

    a.printWolfram();
    println!("");
    b.printWolfram();
    println!("");
    c.printWolfram();
    println!("");

    //Do bw = Bw, then abw = A*(Bw)   
    let mut bw_view : View = View::new(&b, &w, &bw);
    ref_gemm.run( &b, &w, &mut bw, &mut bw_view );
    let mut abw_view : View = View::new(&a, &bw, &abw);
    ref_gemm.run( &a, &bw, &mut abw, &mut abw_view );

    //Do cw = Cw
    let mut cw_view : View = View::new(&c, &w, &cw);
    ref_gemm.run( &c, &w, &mut cw, &mut cw_view );
    
    //Cw -= abw
    axpy( 1.0, &abw, -1.0, &mut cw );
    let norm = cw.frosqr();
    println!("diff is {}", norm);
}
