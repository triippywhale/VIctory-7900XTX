//! `victory warm`: host pipelines made ahead of a run (`known_pipelines`), in
//! processes of their own: a compiler that crashes or hangs costs one pipeline.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use victory_hle::known_pipelines as known;
use victory_shader::emit::host::preferred_vendor;

/// Pipelines a process makes before another takes its place: the compiler's
/// and the translations' memory goes with it.
const AT_ONCE: usize = 40;
/// A pipeline that takes longer is taken for one the compiler hangs on.
const LONGEST: Duration = Duration::from_secs(180);

/// As the runs count one that takes long.
const SLOW_MS: f64 = 1000.0;

/// The processes an AMD driver gives a D3D12 device at one time: in a tenth,
/// `D3D12CreateDevice` fails with 0x887A0005 and the driver reports an error
/// (xcl-dev/xcl 0574950e, Adrenalin 26.11.2 on an RX 9070).
const AMD_DEVICE_PROCESSES: usize = 9;

/// The warm workers by default: `workers`, and on an AMD card (`vendor`
/// 0x1002) no more than leave the title (or another program) one device.
/// `--jobs` is taken as given.
fn with_devices(workers: usize, vendor: Option<u32>) -> usize {
    if vendor == Some(0x1002) {
        workers.min(AMD_DEVICE_PROCESSES - 1)
    } else {
        workers
    }
}

fn folder() -> Result<PathBuf> {
    let cfg = victory_core::Config::load()?;
    Ok(cfg.cache_subdir("shaders")?.with_file_name("pipelines"))
}

pub fn run(jobs: Option<usize>, limit: Option<usize>, again: bool, used: bool, these: Option<String>) -> Result<()> {
    let dir = folder()?;
    if let Some(these) = these {
        return make(&dir, &these);
    }
    let began = Instant::now();
    victory_hle::known_pipelines::use_translations(dir.with_file_name("shaders"));
    let fingerprint = known::fingerprint();
    let done_file = dir.join("warm-done.txt");
    let bad_file = dir.join("warm-bad.txt");
    let keys = known::keys(&dir);
    let listed = std::fs::read_to_string(&done_file).unwrap_or_default();
    let done = recorded_keys(&listed, &fingerprint);
    // The bad ones too are those of this translator and driver: a pipeline
    // the compiler crashed or hung on once was left out for good, though a
    // fix to the translation or a new driver may make it.
    let bad = std::fs::read_to_string(&bad_file).unwrap_or_default();
    let bad = recorded_keys(&bad, &fingerprint);
    let early = std::fs::read_to_string(dir.join("early-pipelines.txt")).unwrap_or_default();
    let early: HashSet<&str> = early.lines().collect();
    // Those earlier runs drew with first, the others in no order (by key).
    let mut all: Vec<usize> = (0..keys.len())
        .filter(|&n| {
            let key = keys[n].as_str();
            (again || !done.contains(key)) && !bad.contains(key) && (!used || early.contains(key))
        })
        .collect();
    all.sort_by_key(|&n| (!early.contains(keys[n].as_str()), &keys[n]));
    all.truncate(limit.unwrap_or(usize::MAX));
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let jobs = jobs
        .unwrap_or_else(|| with_devices(cores * 2 / 3, preferred_vendor()))
        .max(1);
    println!(
        "{} pipelines known, {} made before ({fingerprint}), {} bad; {} to make, {jobs} processes",
        keys.len(),
        done.len(),
        bad.len(),
        all.len()
    );
    let state = Mutex::new((Vec::<f64>::new(), 0usize, 0usize));
    let done_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&done_file)?);
    // (Read back by the runs: these are made beside the others, not before.)
    let slow_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(dir.join("slow-pipelines.txt"))?);
    let bad_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&bad_file)?);
    run_batches(
        &all,
        jobs,
        start_worker,
        |n, ms, there| {
            if ms >= SLOW_MS {
                let _ = writeln!(lock(&slow_out), "\n{} {ms:.0}", keys[n]);
            }
            let mut state = lock(&state);
            state.0.push(ms);
            state.1 += usize::from(!there);
            let count = state.0.len();
            if count.is_multiple_of(1000) {
                println!("{count} of {}, {:.0} s", all.len(), began.elapsed().as_secs_f64());
            }
            drop(state);
            let _ = writeln!(lock(&done_out), "{fingerprint} {}", keys[n]);
        },
        |n, why| {
            println!("pipeline {n} {}: {why}", keys[n]);
            let _ = writeln!(lock(&bad_out), "{fingerprint} {} {why}", keys[n]);
            lock(&state).2 += 1;
        },
    )?;
    let (mut times, refused, crashed) = state.into_inner().unwrap_or_else(|e| e.into_inner());
    times.sort_by(f64::total_cmp);
    let at = |q: f64| times.get((times.len() as f64 * q) as usize).copied().unwrap_or(0.0);
    println!(
        "{} made in {:.0} s ({refused} have no host pipeline, {crashed} ended their process); ms each: median {:.1}, p90 {:.1}, p99 {:.1}, most {:.0}, sum {:.0} s",
        times.len(),
        began.elapsed().as_secs_f64(),
        at(0.5),
        at(0.9),
        at(0.99),
        times.last().copied().unwrap_or(0.0),
        times.iter().sum::<f64>() / 1e3
    );
    Ok(())
}

fn run_batches(
    all: &[usize],
    jobs: usize,
    start: impl Fn(&[usize]) -> Result<Child> + Sync,
    each: impl Fn(usize, f64, bool) + Sync,
    bad: impl Fn(usize, String) + Sync,
) -> Result<()> {
    let chunks: Vec<&[usize]> = all.chunks(AT_ONCE).collect();
    let next = AtomicUsize::new(0);
    // Stop launching workers after a setup failure. In-flight workers can still report.
    let failure = OnceLock::new();
    std::thread::scope(|s| {
        for _ in 0..jobs.min(chunks.len()) {
            s.spawn(|| {
                while let Some(&chunk) = chunks.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let mut rest = chunk;
                    while !rest.is_empty() {
                        if failure.get().is_some() {
                            return;
                        }
                        let outcome = start(rest)
                            .context("could not start warm worker")
                            .and_then(|process| child(process, rest, &each));
                        match outcome {
                            Ok(Some((made, why))) => {
                                let remaining = &rest[made.min(rest.len())..];
                                let Some((&n, after)) = remaining.split_first() else {
                                    break;
                                };
                                bad(n, why);
                                rest = after;
                            }
                            Ok(None) => break,
                            Err(error) => {
                                let _ = failure.set(error);
                                return;
                            }
                        }
                    }
                }
            });
        }
    });
    failure.into_inner().map_or(Ok(()), Err)
}

/// The keys `records` notes under `fingerprint` (the translator with its
/// settings, and the driver): a line is the fingerprint, the key, then
/// anything. Those of another fingerprint, and those of before the lines
/// had one, are made again.
fn recorded_keys<'a>(records: &'a str, fingerprint: &str) -> HashSet<&'a str> {
    records
        .lines()
        .filter_map(|line| {
            let (of, rest) = line.split_once(' ')?;
            if of != fingerprint {
                return None;
            }
            rest.split_ascii_whitespace().next()
        })
        .collect()
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a worker has said so far of the pipelines it was handed: how many
/// it made, and whether it is at one now.
#[derive(Default)]
struct Progress {
    made: usize,
    active: bool,
}

impl Progress {
    /// A line of the worker's (`at <n>`, `made <n> <ms> <0|1>`; anything
    /// else is passed over): what it made, if it says so. A line out of
    /// turn is an error: nothing can be said of the pipelines after it.
    fn record(&mut self, these: &[usize], line: &str) -> Result<Option<(usize, f64, bool)>> {
        let mut words = line.split_ascii_whitespace();
        let kind = match words.next() {
            Some(kind @ ("at" | "made")) => kind,
            _ => return Ok(None),
        };
        let n: usize = words.next().context("missing warm worker index")?.parse()?;
        anyhow::ensure!(these.get(self.made) == Some(&n), "unexpected warm worker index {n}");
        if kind == "at" {
            anyhow::ensure!(!self.active && words.next().is_none(), "invalid warm worker start for {n}");
            self.active = true;
            return Ok(None);
        }
        let ms: f64 = words.next().context("missing warm worker duration")?.parse()?;
        let there = match words.next() {
            Some("0") => false,
            Some("1") => true,
            _ => anyhow::bail!("invalid warm worker result for {n}"),
        };
        anyhow::ensure!(
            self.active && ms.is_finite() && ms >= 0.0 && words.next().is_none(),
            "invalid warm worker completion for {n}"
        );
        self.active = false;
        self.made += 1;
        Ok(Some((n, ms, there)))
    }

    /// The worker ended, badly with `failure`: none if it made all `count`,
    /// or how many it made and why it ended at the next. One that ended
    /// between two pipelines, or well without making them all, is an error:
    /// the pipeline after the last one made had not been started, and was
    /// written down as bad all the same.
    fn finish(self, count: usize, failure: Option<String>) -> Result<Option<(usize, String)>> {
        if let Some(reason) = failure {
            if self.active {
                // It said which pipeline it was making.
                return Ok(Some((self.made, reason)));
            }
            anyhow::bail!("warm worker ended between pipelines after {} of {count}: {reason}", self.made);
        }
        anyhow::ensure!(
            self.made == count,
            "warm worker exited successfully after reporting {} of {count} pipelines",
            self.made
        );
        Ok(None)
    }
}

fn start_worker(these: &[usize]) -> Result<Child> {
    use std::os::windows::process::CommandExt;
    const IDLE_PRIORITY_CLASS: u32 = 0x40;
    let exe = std::env::current_exe().context("could not locate the warm worker executable")?;
    let list: Vec<String> = these.iter().map(usize::to_string).collect();
    Command::new(&exe)
        .args(["warm", "--these", &list.join(",")])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(IDLE_PRIORITY_CLASS)
        .spawn()
        .with_context(|| format!("could not spawn {}", exe.display()))
}

/// Follows a worker making `these`. On abnormal exit, return the number of
/// reports before the failure and the exit reason ([`Progress::finish`]).
fn child(mut process: Child, these: &[usize], mut each: impl FnMut(usize, f64, bool)) -> Result<Option<(usize, String)>> {
    let Some(out) = process.stdout.take() else {
        let _ = process.kill();
        process.wait().context("warm worker stdout unavailable, failed to reap worker")?;
        anyhow::bail!("warm worker stdout unavailable");
    };
    let process = Mutex::new(process);
    let (heard, over) = (Mutex::new(Instant::now()), AtomicBool::new(false));
    let hung = AtomicBool::new(false);
    let mut progress = Progress::default();
    let read = std::thread::scope(|s| {
        s.spawn(|| {
            while !over.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                if lock(&heard).elapsed() > LONGEST {
                    hung.store(true, Ordering::Relaxed);
                    let _ = lock(&process).kill();
                    return;
                }
            }
        });
        let read = (|| -> Result<()> {
            for line in BufReader::new(out).lines() {
                let line = line.context("reading warm worker output")?;
                *lock(&heard) = Instant::now();
                if let Some((n, ms, there)) = progress.record(these, &line)? {
                    each(n, ms, there);
                }
            }
            Ok(())
        })();
        if read.is_err() {
            let _ = lock(&process).kill();
        }
        over.store(true, Ordering::Relaxed);
        read
    });
    let status = lock(&process).wait().context("waiting for the pipeline warm worker")?;
    read?;
    if status.success() {
        return progress.finish(these.len(), None);
    }
    let why = if hung.load(Ordering::Relaxed) { format!("hung for {} s", LONGEST.as_secs()) } else { status.to_string() };
    progress.finish(these.len(), Some(why))
}

/// What such a process does.
fn make(dir: &Path, these: &str) -> Result<()> {
    let these: Vec<usize> = these.split(',').map(|n| n.parse().context("--these")).collect::<Result<_>>()?;
    let cfg = victory_core::Config::load()?;
    victory_hle::known_pipelines::use_translations(cfg.cache_subdir("shaders")?);
    let mut out = std::io::stdout();
    known::make(dir, &these, |n, came| {
        let told = match came {
            None => writeln!(out, "at {n}"),
            Some((ms, there)) => writeln!(out, "made {n} {ms:.2} {}", u8::from(there)),
        };
        if told.and_then(|()| out.flush()).is_err() {
            // Whoever asked is gone.
            std::process::exit(1);
        }
    });
    // Without what a run's end does (nothing to save, the host's threads).
    let _ = out.flush();
    std::process::exit(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::windows::process::CommandExt;

    /// On an AMD card the default leaves one device of the driver's nine for
    /// the title; other vendors keep what the processors give.
    #[test]
    fn warm_workers_on_amd_leave_a_device() {
        assert_eq!(with_devices(16, Some(0x1002)), 8);
        assert_eq!(with_devices(4, Some(0x1002)), 4);
        assert_eq!(with_devices(16, Some(0x10de)), 16);
        assert_eq!(with_devices(16, None), 16);
    }

    fn fake_child(script: &str, stdout: Stdio) -> Result<Child> {
        let cmd = PathBuf::from(std::env::var_os("SystemRoot").unwrap()).join(r"System32\cmd.exe");
        Ok(Command::new(cmd)
            .args(["/d", "/c"])
            .raw_arg(script)
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(Stdio::null())
            .spawn()?)
    }

    #[test]
    fn warm_results_follow_the_translator_and_driver() {
        let records = "current crashed exit code: 101\n\
                       current finished\n\
                       previous slow timed out\n\
                       legacy-key exit code: 101\n\
                       current\n";
        assert_eq!(recorded_keys(records, "current"), HashSet::from(["crashed", "finished"]));
        assert_eq!(recorded_keys(records, "previous"), HashSet::from(["slow"]));
        assert!(recorded_keys(records, "new").is_empty());
    }

    #[test]
    fn only_a_failed_worker_with_a_current_pipeline_marks_it_bad() {
        let chunk = [9, 2, 7];
        let mut p = Progress::default();
        p.record(&chunk, "at 9").unwrap();
        assert_eq!(p.record(&chunk, "made 9 3.00 0").unwrap(), Some((9, 3.0, false)));
        assert!(p.finish(3, Some("exit 101".into())).is_err());

        let mut p = Progress::default();
        p.record(&chunk, "at 9").unwrap();
        p.record(&chunk, "made 9 1.00 1").unwrap();
        p.record(&chunk, "at 2").unwrap();
        assert_eq!(p.finish(3, Some("exit 101".into())).unwrap(), Some((1, "exit 101".into())));
    }

    #[test]
    fn incomplete_or_out_of_order_results_leave_the_rest_retryable() {
        let chunk = [7, 3];
        assert!(Progress::default().finish(2, None).is_err());
        assert!(Progress::default().finish(2, Some("exit 1".into())).is_err());
        let mut p = Progress::default();
        assert!(p.record(&chunk, "made 7 0.1 1").is_err());
        assert!(p.record(&chunk, "at 3").is_err());
        assert!(p.record(&chunk, "diagnostic output").unwrap().is_none());
        p.record(&chunk, "at 7").unwrap();
        assert!(p.record(&chunk, "made 3 0.0 0").is_err());
        assert!(p.record(&chunk, "made 7 0.0 ?").is_err());
        assert!(p.finish(2, None).is_err());
        let mut p = Progress::default();
        for n in chunk {
            p.record(&chunk, &format!("at {n}")).unwrap();
            p.record(&chunk, &format!("made {n} 0.00 0")).unwrap();
        }
        assert_eq!(p.finish(2, None).unwrap(), None);
    }

    /// A worker that ends badly between two pipelines stops the warm-up and
    /// names no pipeline bad: the next one had not been started.
    #[test]
    fn a_worker_ending_between_pipelines_marks_none_bad() {
        let done = Mutex::new(Vec::new());
        let error = run_batches(
            &[7, 8],
            1,
            |_| fake_child("echo at 7&echo made 7 1.0 1&exit /b 7", Stdio::piped()),
            |n, _, _| lock(&done).push(n),
            |_, _| panic!("unexpected bad pipeline"),
        )
        .unwrap_err();
        assert_eq!(*lock(&done), [7]);
        assert!(format!("{error:#}").contains("between pipelines"));
    }

    #[test]
    fn missing_executable_leaves_the_batch_pending() {
        let missing = std::env::temp_dir().join(format!("victory-missing-worker-{}.exe", std::process::id()));
        assert!(!missing.try_exists().unwrap());
        let starts = AtomicUsize::new(0);
        let done = Mutex::new(Vec::new());
        let bad = Mutex::new(Vec::new());
        let result = run_batches(
            &[7, 8, 9],
            1,
            |_| {
                starts.fetch_add(1, Ordering::Relaxed);
                Ok(Command::new(&missing).spawn()?)
            },
            |n, _, _| lock(&done).push(n),
            |n, _| lock(&bad).push(n),
        );
        assert_eq!(starts.load(Ordering::Relaxed), 1);
        assert!(lock(&done).is_empty());
        assert!(lock(&bad).is_empty());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("start warm worker"));
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn missing_stdout_is_a_startup_failure() {
        let error = run_batches(
            &[1, 2],
            1,
            |_| fake_child("exit /b 0", Stdio::null()),
            |_, _, _| panic!("unexpected completion"),
            |_, _| panic!("unexpected bad pipeline"),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("stdout"));
    }

    #[test]
    fn worker_exit_records_the_failed_pipeline_and_resumes() {
        let done = Mutex::new(Vec::new());
        let bad = Mutex::new(Vec::new());
        run_batches(
            &[7, 8, 9],
            1,
            |rest| match rest {
                [7, 8, 9] => fake_child("echo at 7&echo made 7 3.5 1&echo at 8&exit /b 7", Stdio::piped()),
                [9] => fake_child("echo at 9&echo made 9 2.0 0", Stdio::piped()),
                _ => panic!("unexpected batch {rest:?}"),
            },
            |n, ms, there| lock(&done).push((n, ms, there)),
            |n, why| lock(&bad).push((n, why)),
        )
        .unwrap();
        assert_eq!(*lock(&done), [(7, 3.5, true), (9, 2.0, false)]);
        let bad = lock(&bad);
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0].0, 8);
        assert!(!bad[0].1.is_empty());
    }

    #[test]
    fn startup_failure_preserves_another_workers_completion() {
        // Hold the failed start until its peer has reported the only key in its chunk.
        let keys: Vec<_> = (0..=AT_ONCE).collect();
        let (confirmed, received) = std::sync::mpsc::channel();
        let received = Mutex::new(received);
        let error = run_batches(
            &keys,
            2,
            |chunk| {
                if chunk[0] == 0 {
                    assert_eq!(lock(&received).recv_timeout(Duration::from_secs(5)).unwrap(), AT_ONCE);
                    anyhow::bail!("worker unavailable")
                }
                fake_child(&format!("echo at {AT_ONCE}&echo made {AT_ONCE} 1 1"), Stdio::piped())
            },
            |n, _, _| confirmed.send(n).unwrap(),
            |_, _| panic!("unexpected bad pipeline"),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("worker unavailable"));
    }
}
