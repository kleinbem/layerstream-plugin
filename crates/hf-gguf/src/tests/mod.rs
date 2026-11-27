use std::sync::Mutex;

pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());

mod download_and_cache;
mod hub_and_draft;
