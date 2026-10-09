// Run identity. The demo run is the synthetic run that `lab dashboard --demo` streams (events/demo.py).
export const DEMO_RUN_ID = "demo";

export function isDemoRun(runId) {
  return runId === DEMO_RUN_ID;
}

// A signal is demo only when its run is demo. The name of the idea says nothing about the data (FE-06).
export function signalIsDemo(signal) {
  return signal?.demo === true || isDemoRun(signal?.run_id);
}
