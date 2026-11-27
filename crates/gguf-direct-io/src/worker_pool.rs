//! Persistent scoped worker pool that eliminates per-layer OS thread spawn/join overhead.

use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

/// Persistent scoped worker pool that eliminates per-layer OS thread spawn/join overhead.
///
/// The calling thread executes task `0`; `n_tasks - 1` parked background threads execute
/// tasks `1..n_tasks`. `run()` blocks until every background task finishes (even on panic)
/// before returning, so stack borrows remain valid for the entire dispatch.
pub(crate) struct ScopedWorkerPool {
    dispatch_lock: Mutex<()>,
    shared: Arc<PoolShared>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

struct PoolShared {
    state: Mutex<PoolState>,
    work_cv: Condvar,
    done_cv: Condvar,
}

struct PoolState {
    generation: u64,
    active_bg: usize,
    done_bg: usize,
    shutdown: bool,
    task_fn: Option<*const (dyn Fn(usize) + Sync + 'static)>,
}

// SAFETY: `task_fn` points to a `Sync` closure live on the dispatching thread's stack until
// `done_bg == active_bg`; `PoolState` is only accessed under `PoolShared::state`.
unsafe impl Send for PoolState {}

impl ScopedWorkerPool {
    pub(crate) fn new(max_threads: usize) -> Self {
        let bg_count = max_threads.saturating_sub(1);
        let shared = Arc::new(PoolShared {
            state: Mutex::new(PoolState {
                generation: 0,
                active_bg: 0,
                done_bg: 0,
                shutdown: false,
                task_fn: None,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
        });
        let mut handles = Vec::with_capacity(bg_count);
        for _ in 0..bg_count {
            let worker_id = handles.len() + 1;
            let sh = Arc::clone(&shared);
            let h = std::thread::Builder::new()
                .name(format!("ls-io-{worker_id}"))
                .spawn(move || {
                    let mut seen_gen = 0u64;
                    loop {
                        let task_ptr = {
                            let mut st = sh.state.lock();
                            while !st.shutdown && st.generation == seen_gen {
                                sh.work_cv.wait(&mut st);
                            }
                            if st.shutdown {
                                break;
                            }
                            seen_gen = st.generation;
                            if worker_id <= st.active_bg {
                                st.task_fn
                            } else {
                                None
                            }
                        };
                        if let Some(ptr) = task_ptr {
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                // SAFETY: `run()` holds `dispatch_lock` and waits until `done_bg == active_bg`
                                // before clearing `task_fn` and returning, so `ptr` is valid and `Sync`.
                                let f: &(dyn Fn(usize) + Sync) = unsafe { &*ptr };
                                f(worker_id);
                            }));
                            let mut st = sh.state.lock();
                            st.done_bg += 1;
                            if st.done_bg == st.active_bg {
                                sh.done_cv.notify_one();
                            }
                        }
                    }
                });
            if let Ok(handle) = h {
                handles.push(handle);
            }
        }
        Self {
            dispatch_lock: Mutex::new(()),
            shared,
            handles,
        }
    }

    pub(crate) fn run<F: Fn(usize) + Sync>(&self, n_tasks: usize, f: F) {
        if n_tasks == 0 {
            return;
        }
        let max_tasks = self.handles.len() + 1;
        let actual_tasks = n_tasks.min(max_tasks);
        if actual_tasks <= 1 {
            for idx in 0..n_tasks {
                f(idx);
            }
            return;
        }

        let _dispatch_guard = self.dispatch_lock.lock();
        let active_bg = actual_tasks - 1;
        let trait_obj: &(dyn Fn(usize) + Sync) = &f;
        // SAFETY: Erasing the lifetime to `'static` is sound because `WaitGuard` blocks until
        // all `active_bg` background workers finish and clears `task_fn` before `f` is dropped.
        let erased: *const (dyn Fn(usize) + Sync + 'static) =
            unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync), _>(trait_obj) };

        {
            let mut st = self.shared.state.lock();
            st.active_bg = active_bg;
            st.done_bg = 0;
            st.task_fn = Some(erased);
            st.generation = st.generation.wrapping_add(1);
            self.shared.work_cv.notify_all();
        }

        struct WaitGuard<'a>(&'a PoolShared);
        impl Drop for WaitGuard<'_> {
            fn drop(&mut self) {
                let mut st = self.0.state.lock();
                while st.done_bg < st.active_bg {
                    self.0.done_cv.wait(&mut st);
                }
                st.task_fn = None;
                st.active_bg = 0;
            }
        }
        let _wait = WaitGuard(&self.shared);

        // Calling thread executes task 0 (and any overflow tasks if n_tasks > max_tasks).
        f(0);
        for extra in actual_tasks..n_tasks {
            f(extra);
        }
    }
}

impl Drop for ScopedWorkerPool {
    fn drop(&mut self) {
        {
            let mut st = self.shared.state.lock();
            st.shutdown = true;
            self.shared.work_cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
