#![feature(specialization)]
#![feature(alloc, allocator_api)]
#![feature(conservative_impl_trait)]
#![feature(cfg_target_feature)]
#![feature(asm)]
#![feature(iterator_step_by)]

#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![allow(inline_always)] 
#![allow(too_many_arguments)]
#![allow(many_single_char_names)]

extern crate core;
extern crate typenum;
extern crate libc;

pub mod matrix;
pub mod composables;
pub mod thread_comm;
pub mod kern;
pub mod util;
