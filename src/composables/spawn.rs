extern crate scoped_threadpool;
extern crate thread_local;
extern crate hwloc;

use matrix::{Scalar,Mat};
use core::marker::{PhantomData};
use thread_comm::{ThreadComm,ThreadInfo};
use composables::{GemmNode};

use std::sync::{Arc,Mutex};
use std::cell::{RefCell};
use self::scoped_threadpool::Pool;
use self::thread_local::ThreadLocal;
use self::hwloc::{Topology, ObjectType, CPUBIND_THREAD, CpuSet};
use libc;

fn cpuset_for_core(topology: &Topology, idx: usize) -> CpuSet {
    let cores = (*topology).objects_with_type(&ObjectType::Core).unwrap();
    match cores.get(idx) {
        Some(val) => val.cpuset().unwrap(),
        None => panic!("No Core found with id {}", idx)
    }
}

pub struct SpawnThreads<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> 
    where S: Send {
    child: Arc<S>,
    n_threads: usize,
    pool: Pool,

    cntl_cache: Arc<ThreadLocal<RefCell<S>>>,

    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> 
    SpawnThreads <T,At,Bt,Ct,S> 
    where S: Send {
    pub fn new(n_threads: usize, child: S) -> SpawnThreads<T, At, Bt, Ct, S>{
        let mut to_ret = SpawnThreads{ child: Arc::new(child), n_threads : n_threads, pool: Pool::new(n_threads as u32),
                 cntl_cache: Arc::new(ThreadLocal::new()),
                 _t: PhantomData, _at:PhantomData, _bt: PhantomData, _ct: PhantomData };
        to_ret.bind_threads();
        to_ret
    }
    pub fn set_n_threads(&mut self, n_threads: usize){ 
        self.n_threads = n_threads;
        self.pool = Pool::new(n_threads as u32);

        //TODO: Clear control cache more robustly
        Arc::get_mut(&mut self.cntl_cache).expect("").clear();
        self.bind_threads();
    }
    fn bind_threads(&mut self) {
        //Get topology
        let topo = Arc::new(Mutex::new(Topology::new()));
/*        let num_cores = {
            let topo_rc = topo.clone();
            let topo_locked = topo_rc.lock().unwrap();
            (*topo_locked).objects_with_type(&ObjectType::Core).unwrap().len()
        };*/
        let nthr = self.n_threads;
        self.pool.scoped(|scope| {
            for id in 0..nthr {
                let child_topo = topo.clone();
                scope.execute( move || {
                    let tid = unsafe { libc::pthread_self() };
                    {
                    let mut locked_topo = child_topo.lock().unwrap();
//                    let before = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
                    let bind_to = cpuset_for_core(&*locked_topo, id);
                    //Doesn't matter if it worked or not
                    let _ = locked_topo.set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD);
//                    let after = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
    //                    println!("Thread {}: Before {:?}, After {:?}", id, before, after);
                    }
                });
            }
        });
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for SpawnThreads<T, At, Bt, Ct, S> 
    where S: Send {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, _thr: &ThreadInfo<T>) -> () {
        //What we do here:
        //3. Spawn a bunch of threads. How many? 
        //  For now store these in cntl tree.
        //  This will be easy to change since this cntl tree will be the root (so the 'outside' has
        //  ownership.
        //4. Set cpu affinity for each thread.

        //Should the thread communicator be cached???
        //Probably this is cheap so don't worry about it
        let global_comm : Arc<ThreadComm<T>> = Arc::new(ThreadComm::new(self.n_threads));

        //Make some shallow copies here to pass into the scoped,
        //because self.pool borrows self as mutable
        let nthr = self.n_threads;
        let tree_clone = self.child.clone();
        let cache = self.cntl_cache.clone();
    
  
//        println!("Found {} cores.", num_cores);

        self.pool.scoped(|scope| {
            for id in 0..nthr {
                //Make some shallow copies because of borrow rules
                let mut my_a = a.make_alias();
                let mut my_b = b.make_alias();
                let mut my_c = c.make_alias();
                let my_tree = tree_clone.shadow();
                let my_comm  = global_comm.clone();
                let my_cache = cache.clone();
                //let child_topo = topo.clone();

                scope.execute( move || {
                    //Set CPU affinity 
/*                    let tid = unsafe { libc::pthread_self() };
                    {
                    let mut locked_topo = child_topo.lock().unwrap();
                    let before = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
                    let bind_to = cpuset_for_core(&*locked_topo, id);
                    let result = locked_topo.set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD);
                    let after = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
//                    println!("Thread {}: Before {:?}, After {:?}", id, before, after);
                    }*/

                    //Get this thread's control tree
                    let thr = ThreadInfo::new(id, my_comm);
                    let cntl_tree_cell = my_cache.get_or(|| Box::new(RefCell::new(my_tree.shadow())));

                    //Run subproblem
                    cntl_tree_cell.borrow_mut().run(&mut my_a, &mut my_b, &mut my_c, &thr);
                });
            }
        });
    }
    #[inline(always)]
    unsafe fn shadow(&self) -> Self where Self: Sized {
        SpawnThreads{ child: Arc::new(self.child.shadow()), 
                      n_threads : self.n_threads,
                      pool : Pool::new(self.n_threads as u32),
                      cntl_cache: Arc::new(ThreadLocal::new()),
                      _t: PhantomData, _at:PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
