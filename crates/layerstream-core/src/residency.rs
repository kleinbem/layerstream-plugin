//! Global lazy GPU residency manager (0 MB VRAM at idle, auto-evict & model swap).
//!
//! Modeled after Mesh-LLM's `onnx-plugin/src/residency.rs`: all configured models
//! share one active GPU lease so peak VRAM stays strictly bounded and drops back
//! to 0 MB whenever idle.

use std::sync::atomic::{AtomicUsize, Ordering};
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
    activity: Arc<Activity>,
    admission: Arc<Semaphore>,
    idle_timeout: Duration,
    evict_immediately: bool,
}

pub struct EngineLease<E> {
    slot: OwnedMutexGuard<Option<LoadedSlot<E>>>,
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
            let engine = loader()?;
            *self.slot = Some(LoadedSlot {
                model_id: model_id.to_string(),
                engine,
            });
        }
        Ok(&mut self.slot.as_mut().expect("slot just loaded").engine)
    }
}

impl<E> Drop for EngineLease<E> {
    fn drop(&mut self) {
        if self.evict_immediately {
            *self.slot = None;
        }
    }
}

impl<E: Send + 'static> Residency<E> {
    /// Creates a new global lazy GPU residency manager.
    /// When `idle_timeout` is zero, GPU buffers are freed immediately after each request.
    pub fn new(idle_timeout: Duration, max_queued_requests: usize) -> Self {
        Self {
            slot: Arc::new(AsyncMutex::new(None)),
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
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn_blocking(move || {
                    drop(evicted);
                    drop(guard);
                });
            } else {
                drop(evicted);
                drop(guard);
            }
            return true;
        }
        false
    }

    pub fn is_loaded(&self) -> bool {
        self.slot.try_lock().map(|g| g.is_some()).unwrap_or(true)
    }
}
