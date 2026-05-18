/*
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import helium314.keyboard.latin.common.InputPointers;

import java.util.ArrayList;
import java.util.List;

/**
 * Two-thumb point hinter (feature #2.1). Post-processes the {@link InputPointers} aggregated by
 * {@link BatchInputArbiter} just before they're handed to the native glide-typing library. The
 * library API/wire format can't change — but {@link InputPointers} already tags each point with
 * its source {@code pointerId}, so we can reshape what the library sees without touching the JNI
 * boundary.
 *
 * <p>Two cases motivate this class (per HeliBoard issue #291):
 * <ul>
 *   <li><b>Tap-while-swiping (e.g. "firetruck"):</b> one finger glides through {@code fretrc}
 *       while the other taps {@code i}, {@code u}, {@code k}. The native library tends to
 *       underweight short tap-bursts because they look like noise next to the long stroke. We
 *       counteract that by injecting synthetic on-stroke waypoints at each tap's centroid so
 *       the recognizer reads the tap as a deliberate detour on the glide path.</li>
 *   <li><b>Stray opposite-hand tap (e.g. "giraffe" with a left-thumb {@code i}):</b> the tap
 *       looks geometrically out of place and skews the recognizer toward the wrong word. The
 *       midline-based dampener (TODO; not in the MVP) would reduce that tap's influence by
 *       padding its on-path neighbours instead of removing the tap outright (removing risks
 *       losing keys that ONLY that hand types).</li>
 * </ul>
 *
 * <p>The whole hinter is gated by {@code PREF_GESTURE_DUAL_THUMB_HINTING} (default off, marked
 * experimental in the UI). When the pref is off the caller never invokes us — but as a safety
 * net the algorithm also returns the input unchanged whenever fewer than 2 pointers participated
 * or no tap-burst is overlapping a stroke.
 *
 * <p>Conventions:
 * <ul>
 *   <li>All static. No instance state — the algorithm is purely a function of the input.</li>
 *   <li>The {@code input} is never mutated. We return a new {@link InputPointers}.</li>
 *   <li>Caller is expected to be on the main looper at invocation time; we rely on
 *       {@code BatchInputArbiter}'s {@code synchronized (sAggregatedPointers)} ownership of the
 *       buffer for that to be safe.</li>
 * </ul>
 */
public final class DualThumbHinter {
    private DualThumbHinter() {}

    /** Tap classification: total duration ceiling in milliseconds (a tap is brief). */
    private static final int TAP_MAX_DURATION_MS = 80;
    /** Tap classification: max points per pointer-id run (a tap is small). */
    private static final int TAP_MAX_POINTS = 5;
    /**
     * Number of synthetic points injected per detected tap-on-stroke. Three gives the gesture
     * library enough density to treat the tap as a real waypoint without overwhelming the
     * stroke's natural shape. Lower → the tap may still be underweighted; higher → the synthetic
     * pile dominates and can drag the recognizer toward the tap key when the user actually
     * wanted only a faint hint.
     */
    private static final int SYNTHETIC_POINTS_PER_TAP = 3;
    /**
     * Spacing in milliseconds between the synthetic points for a single tap. Keeps them
     * close enough to the tap's real timestamp that they interleave with the stroke's nearby
     * points, while still being a recognizable cluster.
     */
    private static final int SYNTHETIC_POINT_SPACING_MS = 2;
    /**
     * Maximum distance (in multiples of {@code keyWidthPx}) from a tap centroid to the nearest
     * stroke sample for the tap to be considered "on the stroke's trajectory" and therefore
     * eligible for injection. Set conservatively: a stray opposite-hand tap on a key far from
     * the active glide path (the classic "giraffe with stray i" failure mode) is geometrically
     * a detour, and amplifying it would skew the recognizer further. With the guard active,
     * such taps fall through unmodified — the same behaviour the library has today, no
     * regression.
     */
    private static final double PROXIMITY_GATE_KEY_WIDTHS = 1.5;
    /**
     * Within a single pointerId, split into separate logical "runs" whenever consecutive
     * samples are separated by more than this many milliseconds. Multi-finger interleaving
     * usually slots other fingers' samples between same-finger ones — but if the active glide
     * pauses, two distinct taps from the same finger can land adjacent in the aggregate buffer
     * and merge into one bogus "stroke". This gap forces a split.
     */
    private static final int SAME_ID_TIME_GAP_MS = 30;
    /**
     * Within a single pointerId, split into separate logical "runs" whenever consecutive
     * samples are separated by more than this many multiples of {@code keyWidthPx}. Catches the
     * "two distinct taps on different keys, same finger" case that the time gap alone might
     * miss if the user taps in fast succession.
     */
    private static final double SAME_ID_DISTANCE_GAP_KEY_WIDTHS = 1.0;

    /**
     * Result of {@link #postProcess}: the full aggregate to pass to the gesture library, plus
     * just the synthetic injections (for the debug overlay).
     */
    public static final class Result {
        /**
         * Pass this to the gesture library — same content as the {@code raw} input plus any
         * synthetic waypoints, in chronological order. May be the same instance as {@code raw}
         * when no injections were made (cheap fast path).
         */
        @androidx.annotation.NonNull public final InputPointers hinted;
        /**
         * Only the synthetic points (no raw originals). Useful for visualizing what the hinter
         * added on top. Always a fresh, possibly-empty {@link InputPointers}.
         */
        @androidx.annotation.NonNull public final InputPointers syntheticOnly;
        Result(InputPointers hinted, InputPointers syntheticOnly) {
            this.hinted = hinted;
            this.syntheticOnly = syntheticOnly;
        }
    }

    /**
     * Convenience factory: build a no-op {@link Result} that returns {@code raw} unmodified
     * and an empty synthetic-only buffer. Used by callers when they want a uniform return
     * type but the hinter wasn't actually applied (pref off, no geometry, …).
     */
    @androidx.annotation.NonNull
    public static Result identity(@androidx.annotation.NonNull final InputPointers raw) {
        return new Result(raw, EMPTY_SYNTHETIC);
    }

    /** Reused empty synthetic-only buffer for {@link #identity}. Never mutated. */
    private static final InputPointers EMPTY_SYNTHETIC = new InputPointers(0);

    /**
     * Run the hinter. See class docs.
     *
     * @param input the raw aggregated pointers. Not modified.
     * @param keyWidthPx most common key width in pixels (used as the spatial radius for the
     *     "tap stays on one key" classifier). Pass {@code Math.max(keyWidthPx, 1)} to avoid
     *     divide-by-zero if the keyboard hasn't been laid out yet.
     * @param midlineXPx horizontal pixel offset of the left/right hand split. Currently
     *     informational — reserved for the (future) stray-tap dampener.
     * @return a {@link Result} carrying the hinted aggregate AND just the synthetic
     *     injections separately. The hinted aggregate may equal {@code input} if no
     *     injections were made.
     */
    public static Result postProcess(final InputPointers input, final int keyWidthPx,
            @SuppressWarnings("unused") final int midlineXPx) {
        final int n = input.getPointerSize();
        if (n < 2) return new Result(input, new InputPointers(0));
        final int[] xs = input.getXCoordinates();
        final int[] ys = input.getYCoordinates();
        final int[] ids = input.getPointerIds();
        final int[] ts = input.getTimes();

        // Group contiguous same-pointerId points into "runs". Within one gesture batch Android
        // guarantees pointerIds are stable per finger, and BatchInputArbiter feeds events in
        // chronological order — but multi-finger interleaving means the array is not partitioned
        // by id, so we walk and split at every id transition (and at intra-id time/space gaps).
        final List<Run> runs = buildRuns(xs, ys, ids, ts, n, keyWidthPx);
        if (runs.size() < 2) return new Result(input, new InputPointers(0));

        // Classify each run as TAP or STROKE based on time/space extent.
        for (Run r : runs) classify(r, keyWidthPx, xs, ys, ts);

        // Walk runs and accumulate "tap-on-stroke" injections. Each tap that overlaps in time
        // with a stroke spawns a small cluster of synthetic points at the tap's centroid,
        // tagged with the STROKE's pointerId so the library reads them as deliberate waypoints
        // on that stroke. We additionally gate on PROXIMITY: a tap that's geometrically far
        // from the stroke (a stray opposite-hand mis-tap like the "giraffe-with-stray-i" case)
        // is left untouched; amplifying it would worsen recognition rather than improve it.
        final double proximityGatePx = PROXIMITY_GATE_KEY_WIDTHS * Math.max(keyWidthPx, 1);
        final List<SyntheticPoint> injections = new ArrayList<>();
        for (Run tap : runs) {
            if (tap.kind != RunKind.TAP) continue;
            final Run overlappingStroke = findOverlappingStroke(runs, tap, ts);
            if (overlappingStroke == null) continue;
            final int cxRaw = (int)(tap.cumX / Math.max(tap.length, 1));
            final int cyRaw = (int)(tap.cumY / Math.max(tap.length, 1));
            // Proximity guard: only amplify taps that already sit reasonably close to the
            // stroke's trajectory. Distance is measured to the nearest stroke sample, not the
            // closest point on the interpolated path — cheaper and good enough for a guard.
            if (minDistanceToStroke(cxRaw, cyRaw, overlappingStroke, xs, ys) > proximityGatePx) {
                continue;
            }
            // Spread synthetic points around the tap's centre time so they don't all collide
            // on a single timestamp (which the library may dedupe).
            final int baseTime = ts[tap.midIdx];
            for (int k = 0; k < SYNTHETIC_POINTS_PER_TAP; k++) {
                final int offset = (k - SYNTHETIC_POINTS_PER_TAP / 2) * SYNTHETIC_POINT_SPACING_MS;
                injections.add(new SyntheticPoint(
                        cxRaw, cyRaw, overlappingStroke.pointerId, baseTime + offset));
            }
        }

        // TODO(#2.1 follow-up): stray-tap dampener. For each TAP that's geometrically a detour
        // off its overlapping STROKE AND on the opposite side of midlineXPx from the STROKE's
        // dominant side, append duplicates of the stroke's neighbouring on-path points instead
        // (do NOT delete the tap — letters that only that hand types must survive). The
        // proximity guard above only PREVENTS regressing the "giraffe" case; the dampener
        // would actively IMPROVE it.

        if (injections.isEmpty()) return new Result(input, new InputPointers(0));

        // Pre-sort injections by time so we can two-pointer merge into the chronological
        // output stream AND simultaneously stuff a synthetic-only buffer for the debug overlay.
        injections.sort((a, b) -> Integer.compare(a.time, b.time));
        final InputPointers syntheticOnly = new InputPointers(injections.size());
        for (SyntheticPoint sp : injections) {
            syntheticOnly.addPointer(sp.x, sp.y, sp.pointerId, sp.time);
        }
        return new Result(rebuild(xs, ys, ids, ts, n, injections), syntheticOnly);
    }

    /** Minimum Euclidean distance from {@code (px, py)} to any sample of {@code stroke}. */
    private static double minDistanceToStroke(final int px, final int py, final Run stroke,
            final int[] xs, final int[] ys) {
        double best = Double.MAX_VALUE;
        for (int i = stroke.startIdx; i < stroke.endIdx; i++) {
            final double d = Math.hypot(xs[i] - px, ys[i] - py);
            if (d < best) best = d;
        }
        return best;
    }

    // ---- internals ----

    /** Pointer-run kinds. {@link #UNKNOWN} only exists for partially-classified runs in tests. */
    private enum RunKind { UNKNOWN, TAP, STROKE }

    /**
     * One contiguous same-pointerId slice of the input array. Indices are into the original
     * {@code xs/ys/ids/ts} arrays.
     */
    private static final class Run {
        final int pointerId;
        final int startIdx; // inclusive
        final int endIdx;   // exclusive
        final int length;
        int midIdx;         // index of the temporal midpoint, used for centroid sampling
        long cumX;          // sum of x for centroid; long to avoid overflow on large gestures
        long cumY;
        RunKind kind = RunKind.UNKNOWN;

        Run(int pointerId, int startIdx, int endIdx) {
            this.pointerId = pointerId;
            this.startIdx = startIdx;
            this.endIdx = endIdx;
            this.length = endIdx - startIdx;
            this.midIdx = startIdx + length / 2;
        }
    }

    /**
     * A synthetic waypoint to be spliced into the output stream in time order.
     */
    private static final class SyntheticPoint {
        final int x, y, pointerId, time;
        SyntheticPoint(int x, int y, int pointerId, int time) {
            this.x = x;
            this.y = y;
            this.pointerId = pointerId;
            this.time = time;
        }
    }

    /**
     * Split the input into runs that are both same-pointerId AND temporally/spatially contiguous.
     * The second condition matters because the multi-finger aggregation in
     * {@code BatchInputArbiter} can splice consecutive same-finger taps next to each other in
     * the array when the OTHER finger isn't producing samples in between. Two physically
     * distinct taps merged into one logical "run" would yield a bogus centroid and probably be
     * misclassified as a stroke; the gap-split fixes that.
     */
    private static List<Run> buildRuns(int[] xs, int[] ys, int[] ids, int[] ts, int n,
            final int keyWidthPx) {
        final double sameIdDistanceGapPx = SAME_ID_DISTANCE_GAP_KEY_WIDTHS * Math.max(keyWidthPx, 1);
        final List<Run> runs = new ArrayList<>();
        int runStart = 0;
        for (int i = 1; i < n; i++) {
            boolean splitHere = false;
            if (ids[i] != ids[runStart]) {
                splitHere = true;
            } else {
                // Same pointerId — only split if there's a real discontinuity. We compare to
                // the IMMEDIATELY-PRECEDING same-id sample (i-1 may belong to another pointer,
                // which is fine; the time/space gap measured against i-1 still captures "a long
                // time with no same-finger activity" because BatchInputArbiter appends the
                // other finger's samples in chronological order.)
                final int prevSameIdIdx = findPrevSameId(ids, runStart, i);
                if (prevSameIdIdx >= 0) {
                    final int dt = ts[i] - ts[prevSameIdIdx];
                    final double dist = Math.hypot(xs[i] - xs[prevSameIdIdx],
                            ys[i] - ys[prevSameIdIdx]);
                    if (dt > SAME_ID_TIME_GAP_MS || dist > sameIdDistanceGapPx) {
                        splitHere = true;
                    }
                }
            }
            if (splitHere) {
                runs.add(new Run(ids[runStart], runStart, i));
                runStart = i;
            }
        }
        runs.add(new Run(ids[runStart], runStart, n));
        return runs;
    }

    /** Return the index of the most recent same-pointerId sample in [{@code lo}, {@code hi}), or -1. */
    private static int findPrevSameId(final int[] ids, final int lo, final int hi) {
        final int target = ids[hi];
        for (int j = hi - 1; j >= lo; j--) {
            if (ids[j] == target) return j;
        }
        return -1;
    }

    /**
     * Classify a {@link Run} as TAP or STROKE based on its temporal and spatial extent, and
     * compute its centroid sums + midpoint index along the way.
     */
    private static void classify(final Run r, final int keyWidthPx,
            final int[] xs, final int[] ys, final int[] ts) {
        if (r.length == 0) {
            r.kind = RunKind.STROKE; // degenerate — treat as not-a-tap so we don't synthesize
            return;
        }
        int minX = xs[r.startIdx], maxX = minX;
        int minY = ys[r.startIdx], maxY = minY;
        long sumX = 0, sumY = 0;
        for (int i = r.startIdx; i < r.endIdx; i++) {
            final int x = xs[i], y = ys[i];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            sumX += x;
            sumY += y;
        }
        r.cumX = sumX;
        r.cumY = sumY;
        final int durationMs = ts[r.endIdx - 1] - ts[r.startIdx];
        // Diagonal radius — covers worst-case spread regardless of vertical vs horizontal extent.
        final double radius = Math.hypot(maxX - minX, maxY - minY);
        final boolean isTap = durationMs <= TAP_MAX_DURATION_MS
                && r.length <= TAP_MAX_POINTS
                && radius <= Math.max(keyWidthPx, 1) / 2.0;
        r.kind = isTap ? RunKind.TAP : RunKind.STROKE;
    }

    /**
     * Find a STROKE run whose time span overlaps the given TAP's time span. Returns the first
     * such stroke (gestures rarely have more than two concurrent strokes; for >2 fingers we
     * just attribute the tap to the first overlapping stroke, which is the simplest defensible
     * behaviour). {@code null} if none overlap (a "free" tap — e.g. after the only stroke ended,
     * or before any stroke began).
     */
    private static Run findOverlappingStroke(final List<Run> runs, final Run tap,
            final int[] ts) {
        final int tapStartTime = ts[tap.startIdx];
        final int tapEndTime = ts[tap.endIdx - 1];
        for (Run r : runs) {
            if (r == tap || r.kind != RunKind.STROKE || r.length == 0) continue;
            if (r.pointerId == tap.pointerId) continue; // same finger as the tap — not a sibling
            // Two time-intervals overlap iff neither ends before the other begins. Inclusive
            // endpoints are fine here: a tap that's a single sample at the exact moment the
            // stroke also has a sample still counts as concurrent.
            final int strokeStartTime = ts[r.startIdx];
            final int strokeEndTime = ts[r.endIdx - 1];
            if (strokeEndTime < tapStartTime) continue; // stroke ended before tap started
            if (strokeStartTime > tapEndTime) continue; // stroke started after tap ended
            return r;
        }
        return null;
    }

    /**
     * Build a new {@link InputPointers} containing the original samples plus the synthetic
     * injections, in chronological order (stable: original samples keep their relative order
     * when timestamps tie; synthetic points sort after originals with the same timestamp).
     * {@code injections} must already be sorted ascending by {@code time} — the caller does
     * that so the synthetic-only buffer can be populated in the same pass.
     */
    private static InputPointers rebuild(final int[] xs, final int[] ys, final int[] ids,
            final int[] ts, final int n, final List<SyntheticPoint> injections) {
        final int outSize = n + injections.size();
        final InputPointers out = new InputPointers(outSize);
        int oi = 0; // original index
        int ji = 0; // injection index
        while (oi < n && ji < injections.size()) {
            final SyntheticPoint s = injections.get(ji);
            if (ts[oi] <= s.time) {
                out.addPointer(xs[oi], ys[oi], ids[oi], ts[oi]);
                oi++;
            } else {
                out.addPointer(s.x, s.y, s.pointerId, s.time);
                ji++;
            }
        }
        while (oi < n) {
            out.addPointer(xs[oi], ys[oi], ids[oi], ts[oi]);
            oi++;
        }
        while (ji < injections.size()) {
            final SyntheticPoint s = injections.get(ji);
            out.addPointer(s.x, s.y, s.pointerId, s.time);
            ji++;
        }
        return out;
    }
}
