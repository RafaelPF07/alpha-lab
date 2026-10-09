// A throttle that never drops the last request. The first request runs at once. A request inside the interval
// is held back, and one trailing run happens when the interval ends, so the newest state always reaches the page
// (R4-FE-06). A forced request runs at once and cancels the pending one.
//
// clock and timers are injectable so the tests can drive time by hand. The defaults are the browser's.
export function trailingThrottle(fn, intervalMs, {
  now = () => Date.now(),
  schedule = (callback, ms) => setTimeout(callback, ms),
  cancel = (handle) => clearTimeout(handle),
} = {}) {
  let last = -Infinity;
  let pending = null;

  const run = (at) => {
    pending = null;
    last = at;
    fn(at);
  };

  return {
    // Runs fn now when the interval has passed or force is true. Otherwise it arranges one trailing run.
    request(force = false) {
      const at = now();
      if (force) {
        if (pending !== null) cancel(pending);
        pending = null;
        run(at);
        return;
      }
      if (at - last >= intervalMs) {
        run(at);
        return;
      }
      if (pending === null) {
        pending = schedule(() => run(now()), Math.max(0, last + intervalMs - at));
      }
    },

    // Forgets the last run and drops any pending one, so the next request runs at once (a new run).
    reset() {
      if (pending !== null) cancel(pending);
      pending = null;
      last = -Infinity;
    },
  };
}
