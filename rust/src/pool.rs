use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread;

struct JobDesc {
    func: fn(*const (), usize) -> i32,
    ctx: usize,
    n_tasks: usize,
    active_workers: usize,
    gen: usize,
}

struct PoolState {
    job: Option<JobDesc>,
    done_workers: usize,
}

pub struct GlobalPool {
    state: Mutex<PoolState>,
    cv_work: Condvar,
    cv_done: Condvar,
    next_task: AtomicUsize,
    err_code: AtomicI32,
    num_threads: usize,
}

static POOL: OnceLock<GlobalPool> = OnceLock::new();

impl GlobalPool {
    fn get() -> &'static Self {
        static INIT_THREADS: OnceLock<()> = OnceLock::new();
        let pool = POOL.get_or_init(|| GlobalPool {
            state: Mutex::new(PoolState {
                job: None,
                done_workers: 0,
            }),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
            next_task: AtomicUsize::new(0),
            err_code: AtomicI32::new(0),
            num_threads: 16,
        });
        INIT_THREADS.get_or_init(|| {
            for tid in 0..pool.num_threads {
                thread::spawn(move || pool.worker_loop(tid));
            }
        });
        pool
    }

    fn worker_loop(&'static self, tid: usize) {
        let mut last_gen = 0usize;
        loop {
            let (func, ctx, n_tasks) = {
                let mut g = self.state.lock().unwrap();
                loop {
                    if let Some(ref job) = g.job {
                        if job.gen != last_gen && tid < job.active_workers {
                            last_gen = job.gen;
                            break (job.func, job.ctx, job.n_tasks);
                        }
                    }
                    g = self.cv_work.wait(g).unwrap();
                }
            };

            self.run_tasks(func, ctx as *const (), n_tasks);

            let mut g = self.state.lock().unwrap();
            g.done_workers += 1;
            if let Some(ref job) = g.job {
                if g.done_workers == job.active_workers {
                    self.cv_done.notify_one();
                }
            }
        }
    }

    #[inline]
    fn run_tasks(&self, func: fn(*const (), usize) -> i32, ctx: *const (), n_tasks: usize) {
        loop {
            if self.err_code.load(Ordering::Relaxed) != 0 {
                break;
            }
            let task_id = self.next_task.fetch_add(1, Ordering::Relaxed);
            if task_id >= n_tasks {
                break;
            }
            let rc = func(ctx, task_id);
            if rc != 0 {
                self.err_code.store(rc, Ordering::Relaxed);
                break;
            }
        }
    }
}

static RUN_LOCK: Mutex<usize> = Mutex::new(0);

pub fn par_for_each<F>(n_tasks: usize, n_workers: usize, body: F) -> i32
where
    F: Fn(usize) -> i32 + Sync,
{
    if n_tasks == 0 {
        return 0;
    }
    let workers = n_workers.min(n_tasks);
    if workers <= 1 {
        for i in 0..n_tasks {
            let rc = body(i);
            if rc != 0 {
                return rc;
            }
        }
        return 0;
    }

    fn trampoline<F: Fn(usize) -> i32 + Sync>(ctx: *const (), task_id: usize) -> i32 {
        let f = unsafe { &*(ctx as *const F) };
        f(task_id)
    }

    let pool = GlobalPool::get();
    let mut gen_guard = RUN_LOCK.lock().unwrap();
    *gen_guard = gen_guard.wrapping_add(1);
    let gen = if *gen_guard == 0 {
        *gen_guard = 1;
        1
    } else {
        *gen_guard
    };

    let bg_workers = (workers - 1).min(pool.num_threads);
    pool.next_task.store(0, Ordering::Relaxed);
    pool.err_code.store(0, Ordering::Relaxed);

    if bg_workers > 0 {
        let mut g = pool.state.lock().unwrap();
        g.done_workers = 0;
        g.job = Some(JobDesc {
            func: trampoline::<F>,
            ctx: (&body as *const F) as usize,
            n_tasks,
            active_workers: bg_workers,
            gen,
        });
        drop(g);
        pool.cv_work.notify_all();
    }

    pool.run_tasks(trampoline::<F>, (&body as *const F) as *const (), n_tasks);

    if bg_workers > 0 {
        let mut g = pool.state.lock().unwrap();
        while g.done_workers < bg_workers {
            g = pool.cv_done.wait(g).unwrap();
        }
        g.job = None;
    }

    pool.err_code.load(Ordering::Relaxed)
}
