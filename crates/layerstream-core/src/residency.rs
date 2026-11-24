//! Global lazy GPU residency manager (0 MB VRAM at idle, auto-evict & model swap).
//!
//! Modeled after Mesh-LLM's `onnx-plugin/src/residency.rs`: all configured models
//! share one active GPU lease so peak VRAM stays strictly bounded and drops back
//! to 0 MB whenever idle.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard, OwnedSemaphorePermit, Semaphore};

struct LoadedSlot<E> {
    model_id: String,
    engine: E,
}

struct Activity {
    in_flight: AtomicUsize,
    last_finished: Mutex<Instant>,
}

struct RequestLease {
    activity: Arc<Activity>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for RequestLease {
    fn drop(&mut self) {
        if let Ok(mut ts) = self.activity.last_finished.lock() {
            *ts = Instant::now();
        }
        self.activity.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct AdmissionPermit {
    request: RequestLease,
}

pub struct Residency<E> {
    slot: Arc<AsyncMutex<Option<LoadedSlot<E>>>>,
    loaded: Arc<AtomicBool>,
    activity: Arc<Activity>,
    admission: Arc<Semaphore>,
    idle_timeout: Duration,
    evict_immediately: bool,
}

pub struct EngineLease<E> {
    slot: OwnedMutexGuard<Option<LoadedSlot<E>>>,
    loaded: Arc<AtomicBool>,
    evict_immediately: bool,
    _request: RequestLease,
}

impl<E: Send + 'static> EngineLease<E> {
    pub fn model_id(&self) -> Option<&str> {
        self.slot.as_ref().map(|s| s.model_id.as_str())
    }

    pub fn get_or_load<F>(&mut self, model_id: &str, loader: F) -> Result<&mut E>
    where
        F: FnOnce() -> Result<E>,
    {
        let needs_load = match self.slot.as_ref() {
            Some(existing) => existing.model_id != model_id,
            None => true,
        };
        if needs_load {
            // Drop previous model FIRST so its GPU buffers are freed before allocating the next one.
            *self.slot = None;
            self.loaded.store(false, Ordering::SeqCst);
            let engine = loader()?;
            *self.slot = Some(LoadedSlot {
                model_id: model_id.to_string(),
                engine,
            });
            self.loaded.store(true, Ordering::SeqCst);
        }
        Ok(&mut self.slot.as_mut().expect("slot just loaded").engine)
    }
}

impl<E> Drop for EngineLease<E> {
    fn drop(&mut self) {
        if self.evict_immediately {
            *self.slot = None;
            self.loaded.store(false, Ordering::SeqCst);
        }
    }
}

impl<E: Send + 'static> Residency<E> {
    /// Creates a new global lazy GPU residency manager.
    /// When `idle_timeout` is zero, GPU buffers are freed immediately after each request.
    pub fn new(idle_timeout: Duration, max_queued_requests: usize) -> Self {
        Self {
            slot: Arc::new(AsyncMutex::new(None)),
            loaded: Arc::new(AtomicBool::new(false)),
            activity: Arc::new(Activity {
                in_flight: AtomicUsize::new(0),
                last_finished: Mutex::new(Instant::now()),
            }),
            admission: Arc::new(Semaphore::new(max_queued_requests.max(1))),
            idle_timeout,
            evict_immediately: idle_timeout.is_zero(),
        }
    }

    /// Synchronously claims a queue admission slot before committing HTTP 200 SSE headers.
    /// Returns an error immediately if `max_queued_requests` is saturated.
    pub fn try_acquire_admission(&self) -> Result<AdmissionPermit> {
        let permit = self
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| anyhow!("layerstream request queue is full"))?;
        self.activity.in_flight.fetch_add(1, Ordering::SeqCst);
        Ok(AdmissionPermit {
            request: RequestLease {
                activity: self.activity.clone(),
                _permit: permit,
            },
        })
    }

    /// Waits for exclusive engine ownership using a previously claimed admission permit.
    pub async fn lock_with_permit(&self, permit: AdmissionPermit) -> EngineLease<E> {
        let slot = self.slot.clone().lock_owned().await;
        EngineLease {
            slot,
            loaded: self.loaded.clone(),
            evict_immediately: self.evict_immediately,
            _request: permit.request,
        }
    }

    pub async fn acquire(&self) -> Result<EngineLease<E>> {
        let permit = self.try_acquire_admission()?;
        Ok(self.lock_with_permit(permit).await)
    }

    /// Evicts any idle loaded model if `idle_timeout` has elapsed with zero in-flight requests.
    /// Drops the evicted engine on a blocking worker thread when inside a Tokio runtime so
    /// `worker_thread.join()` and multi-GiB GPU frees never stall async I/O workers.
    pub fn evict_idle(&self, now: Instant) -> bool {
        let Ok(mut guard) = self.slot.clone().try_lock_owned() else {
            return false;
        };
        if guard.is_none() || self.activity.in_flight.load(Ordering::SeqCst) != 0 {
            return false;
        }
        let Ok(last_finished) = self.activity.last_finished.lock() else {
            return false;
        };
        if now.saturating_duration_since(*last_finished) < self.idle_timeout {
            return false;
        }
        if let Some(evicted) = guard.take() {
            let loaded = self.loaded.clone();
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn_blocking(move || {
                    drop(evicted);
                    loaded.store(false, Ordering::SeqCst);
                    drop(guard);
                });
            } else {
                drop(evicted);
                loaded.store(false, Ordering::SeqCst);
                drop(guard);
            }
            return true;
        }
        false
    }

    /// Unconditionally evicts any loaded model once exclusive slot ownership is acquired.
    /// Drops the engine on a blocking worker thread so multi-GiB GPU frees never stall async I/O.
    pub async fn evict_now(&self) -> bool {
        let mut guard = self.slot.clone().lock_owned().await;
        let Some(evicted) = guard.take() else {
            return false;
        };
        let loaded = self.loaded.clone();
        let _ = tokio::task::spawn_blocking(move || {
            drop(evicted);
            loaded.store(false, Ordering::SeqCst);
            drop(guard);
        })
        .await;
        true
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DropTracker {
        id: &'static str,
        events: Arc<Mutex<Vec<String>>>,
    }

    impl Drop for DropTracker {
        fn drop(&mut self) {
            self.events
                .lock()
                .unwrap()
                .push(format!("drop:{}", self.id));
        }
    }

    #[tokio::test]
    async fn model_swap_drops_previous_before_loading_next() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let res: Residency<DropTracker> = Residency::new(Duration::from_secs(60), 2);

        {
            let mut lease = res.acquire().await.unwrap();
            let ev = events.clone();
            lease
                .get_or_load("m1", move || {
                    ev.lock().unwrap().push("load:m1".to_string());
                    Ok(DropTracker {
                        id: "m1",
                        events: ev,
                    })
                })
                .unwrap();
            assert_eq!(lease.model_id(), Some("m1"));
            assert!(res.is_loaded());
        }

        {
            let mut lease = res.acquire().await.unwrap();
            let ev = events.clone();
            lease
                .get_or_load("m2", move || {
                    ev.lock().unwrap().push("load:m2".to_string());
                    Ok(DropTracker {
                        id: "m2",
                        events: ev,
                    })
                })
                .unwrap();
            assert_eq!(lease.model_id(), Some("m2"));
        }

        assert_eq!(
            *events.lock().unwrap(),
            vec!["load:m1", "drop:m1", "load:m2"]
        );
        assert!(res.evict_now().await);
        assert!(!res.is_loaded());
        assert!(!res.evict_now().await);
    }

    #[tokio::test]
    async fn immediate_evict_and_admission_limit_enforced() {
        let res: Residency<u32> = Residency::new(Duration::ZERO, 1);
        let p1 = res.try_acquire_admission().unwrap();
        assert!(res.try_acquire_admission().is_err());

        {
            let mut lease = res.lock_with_permit(p1).await;
            lease.get_or_load("m1", || Ok(42)).unwrap();
            assert!(res.is_loaded());
        }
        assert!(!res.is_loaded());
        assert!(res.try_acquire_admission().is_ok());
    }

    #[tokio::test]
    async fn idle_eviction_waits_for_timeout_and_respects_in_flight() {
        let res: Residency<u32> = Residency::new(Duration::from_secs(10), 2);
        {
            let mut lease = res.acquire().await.unwrap();
            lease.get_or_load("m1", || Ok(7)).unwrap();
            assert!(!res.evict_idle(Instant::now() + Duration::from_secs(60)));
        }

        assert!(!res.evict_idle(Instant::now()));
        assert!(res.is_loaded());
        assert!(res.evict_idle(Instant::now() + Duration::from_secs(15)));
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!res.is_loaded());
    }
}
