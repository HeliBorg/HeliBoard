/*
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import androidx.annotation.NonNull;

import helium314.keyboard.keyboard.PointerTracker;
import helium314.keyboard.latin.common.InputPointers;

/**
 * Visual debug overlay (feature #2.1). Draws the points the gesture library actually sees,
 * superimposed on the keyboard. Helpful when iterating on {@link DualThumbHinter} and similar
 * point-shaping experiments — the user can toggle {@code PREF_GESTURE_DEBUG_DRAW_POINTS} on,
 * gesture a word, and visually inspect raw vs. processed samples.
 *
 * <p>The overlay snapshots the inputs at each batch-end (immediate or grace-deferred) and keeps
 * them visible until the next batch starts, so the trail is still on screen when the user
 * compares it with the suggestion strip. Two colour bands distinguish the streams:
 * <ul>
 *   <li>raw samples → small red dots, semi-transparent (60 % opaque),</li>
 *   <li>synthetic samples added by {@link DualThumbHinter} → slightly larger blue dots, fully
 *       opaque so they pop against the raw points.</li>
 * </ul>
 *
 * <p>Everything is single-threaded on the UI / main-looper thread: {@link #updateSnapshot} is
 * called from {@link PointerTracker}'s commit paths (touch-event or {@code Handler} dispatch,
 * both on main), and {@link #drawPreview} is called from the {@code DrawingPreviewPlacerView}'s
 * {@code onDraw} which is also main-thread. No synchronisation is required as long as that
 * invariant holds.
 */
public final class GestureDebugPointsDrawingPreview extends AbstractDrawingPreview {
    // Snapshot of the most recent batch's points. Held as primitive int[] copies so the
    // upstream BatchInputArbiter is free to reset its aggregate without affecting us.
    private int[] mRawXs;
    private int[] mRawYs;
    private int[] mSyntheticXs;
    private int[] mSyntheticYs;

    private final Paint mRawPaint = new Paint();
    private final Paint mSyntheticPaint = new Paint();
    /** Radius in pixels for a "raw" sample dot. Picked so dots remain visible at typical DPIs. */
    private static final float RAW_RADIUS_PX = 4f;
    /** Radius for "synthetic" sample dots — deliberately larger so the diff is obvious. */
    private static final float SYNTHETIC_RADIUS_PX = 7f;

    public GestureDebugPointsDrawingPreview() {
        mRawPaint.setAntiAlias(true);
        mRawPaint.setColor(Color.RED);
        mRawPaint.setAlpha(0x99); // 60 % opaque — raw stream is dense, so visually a wash
        mRawPaint.setStyle(Paint.Style.FILL);

        mSyntheticPaint.setAntiAlias(true);
        mSyntheticPaint.setColor(Color.BLUE);
        mSyntheticPaint.setAlpha(0xFF);
        mSyntheticPaint.setStyle(Paint.Style.FILL);
    }

    /**
     * Replace the on-screen snapshot. Both arguments are required; pass an empty
     * {@link InputPointers} for {@code synthetic} when the hinter isn't active (the raw stream
     * is still useful on its own). The arrays inside {@link InputPointers} are mutated by the
     * arbiter on each gesture, so we make defensive copies up to {@code rawSize} /
     * {@code syntheticSize}.
     *
     * @param raw the unprocessed pointers as the keyboard layer produced them.
     * @param synthetic only the additional points injected on top of {@code raw}; if the
     *     hinter is disabled or didn't add anything, this should be an empty
     *     {@link InputPointers} or {@code null} (treated identically — nothing blue drawn).
     */
    public void updateSnapshot(@NonNull final InputPointers raw, final InputPointers synthetic) {
        mRawXs = copyOfLength(raw.getXCoordinates(), raw.getPointerSize());
        mRawYs = copyOfLength(raw.getYCoordinates(), raw.getPointerSize());
        if (synthetic != null && synthetic.getPointerSize() > 0) {
            mSyntheticXs = copyOfLength(synthetic.getXCoordinates(), synthetic.getPointerSize());
            mSyntheticYs = copyOfLength(synthetic.getYCoordinates(), synthetic.getPointerSize());
        } else {
            mSyntheticXs = null;
            mSyntheticYs = null;
        }
        invalidateDrawingView();
    }

    /** Drop the overlay (e.g. on gesture cancel or view teardown). */
    public void clear() {
        mRawXs = null;
        mRawYs = null;
        mSyntheticXs = null;
        mSyntheticYs = null;
        invalidateDrawingView();
    }

    @Override
    public void onDeallocateMemory() {
        clear();
    }

    @Override
    public void drawPreview(@NonNull final Canvas canvas) {
        if (!isPreviewEnabled()) return;
        // Raw first so synthetic dots draw on top.
        drawPoints(canvas, mRawXs, mRawYs, mRawPaint, RAW_RADIUS_PX);
        drawPoints(canvas, mSyntheticXs, mSyntheticYs, mSyntheticPaint, SYNTHETIC_RADIUS_PX);
    }

    @Override
    public void setPreviewPosition(@NonNull final PointerTracker tracker) {
        // No-op. The overlay is fed externally via {@link #updateSnapshot}; it doesn't track
        // a single pointer the way the trail / floating-text previews do.
    }

    private static void drawPoints(final Canvas canvas, final int[] xs, final int[] ys,
            final Paint paint, final float radiusPx) {
        if (xs == null || ys == null) return;
        final int n = Math.min(xs.length, ys.length);
        for (int i = 0; i < n; i++) {
            canvas.drawCircle(xs[i], ys[i], radiusPx, paint);
        }
    }

    private static int[] copyOfLength(final int[] src, final int length) {
        if (length <= 0) return null;
        final int[] copy = new int[length];
        System.arraycopy(src, 0, copy, 0, length);
        return copy;
    }
}
