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
 * compares it with the suggestion strip. The overlay distinguishes the streams and gesture
 * structure:
 * <ul>
 *   <li>raw samples and connecting segments are colour-coded by pointer/finger id,</li>
 *   <li>short tap-like runs are outlined as squares,</li>
 *   <li>each pointer run has a green start ring and black end marker,</li>
 *   <li>synthetic samples added by {@link DualThumbHinter} are drawn as blue crosses.</li>
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
    private int[] mRawIds;
    private int[] mRawTimes;
    private int[] mSyntheticXs;
    private int[] mSyntheticYs;
    private int[] mSyntheticIds;
    private int[] mSyntheticTimes;

    private final Paint mRawPaint = new Paint();
    private final Paint mLinePaint = new Paint();
    private final Paint mSyntheticPaint = new Paint();
    private final Paint mStartPaint = new Paint();
    private final Paint mEndPaint = new Paint();
    private final Paint mTapPaint = new Paint();
    /** Radius in pixels for a "raw" sample dot. Picked so dots remain visible at typical DPIs. */
    private static final float RAW_RADIUS_PX = 4f;
    private static final float SYNTHETIC_CROSS_RADIUS_PX = 8f;
    private static final float TAP_MARKER_RADIUS_PX = 10f;
    private static final float START_MARKER_RADIUS_PX = 9f;
    private static final float END_MARKER_RADIUS_PX = 8f;
    private static final int TAP_MAX_POINTS = 5;
    private static final int TAP_MAX_DURATION_MS = 80;
    private static final int[] POINTER_COLORS = {
            Color.rgb(244, 67, 54),   // red
            Color.rgb(76, 175, 80),   // green
            Color.rgb(255, 152, 0),   // orange
            Color.rgb(156, 39, 176),  // purple
            Color.rgb(0, 188, 212),   // cyan
            Color.rgb(255, 235, 59),  // yellow
    };

    public GestureDebugPointsDrawingPreview() {
        mRawPaint.setAntiAlias(true);
        mRawPaint.setAlpha(0x99); // 60 % opaque — raw stream is dense, so visually a wash
        mRawPaint.setStyle(Paint.Style.FILL);

        mLinePaint.setAntiAlias(true);
        mLinePaint.setAlpha(0x77);
        mLinePaint.setStrokeWidth(3f);
        mLinePaint.setStyle(Paint.Style.STROKE);

        mSyntheticPaint.setAntiAlias(true);
        mSyntheticPaint.setColor(Color.BLUE);
        mSyntheticPaint.setAlpha(0xFF);
        mSyntheticPaint.setStrokeWidth(4f);
        mSyntheticPaint.setStyle(Paint.Style.STROKE);

        mStartPaint.setAntiAlias(true);
        mStartPaint.setColor(Color.rgb(0, 200, 83));
        mStartPaint.setAlpha(0xFF);
        mStartPaint.setStrokeWidth(4f);
        mStartPaint.setStyle(Paint.Style.STROKE);

        mEndPaint.setAntiAlias(true);
        mEndPaint.setColor(Color.BLACK);
        mEndPaint.setAlpha(0xCC);
        mEndPaint.setStyle(Paint.Style.FILL);

        mTapPaint.setAntiAlias(true);
        mTapPaint.setColor(Color.WHITE);
        mTapPaint.setAlpha(0xDD);
        mTapPaint.setStrokeWidth(3f);
        mTapPaint.setStyle(Paint.Style.STROKE);
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
        mRawIds = copyOfLength(raw.getPointerIds(), raw.getPointerSize());
        mRawTimes = copyOfLength(raw.getTimes(), raw.getPointerSize());
        if (synthetic != null && synthetic.getPointerSize() > 0) {
            mSyntheticXs = copyOfLength(synthetic.getXCoordinates(), synthetic.getPointerSize());
            mSyntheticYs = copyOfLength(synthetic.getYCoordinates(), synthetic.getPointerSize());
            mSyntheticIds = copyOfLength(synthetic.getPointerIds(), synthetic.getPointerSize());
            mSyntheticTimes = copyOfLength(synthetic.getTimes(), synthetic.getPointerSize());
        } else {
            mSyntheticXs = null;
            mSyntheticYs = null;
            mSyntheticIds = null;
            mSyntheticTimes = null;
        }
        invalidateDrawingView();
    }

    /** Drop the overlay (e.g. on gesture cancel or view teardown). */
    public void clear() {
        mRawXs = null;
        mRawYs = null;
        mRawIds = null;
        mRawTimes = null;
        mSyntheticXs = null;
        mSyntheticYs = null;
        mSyntheticIds = null;
        mSyntheticTimes = null;
        invalidateDrawingView();
    }

    @Override
    public void onDeallocateMemory() {
        clear();
    }

    @Override
    public void drawPreview(@NonNull final Canvas canvas) {
        if (!isPreviewEnabled()) return;
        drawPointerLines(canvas);
        drawRawPoints(canvas);
        drawRunMarkers(canvas);
        drawSyntheticPoints(canvas);
    }

    @Override
    public void setPreviewPosition(@NonNull final PointerTracker tracker) {
        // No-op. The overlay is fed externally via {@link #updateSnapshot}; it doesn't track
        // a single pointer the way the trail / floating-text previews do.
    }

    private void drawPointerLines(final Canvas canvas) {
        if (mRawXs == null || mRawYs == null || mRawIds == null) return;
        final int n = Math.min(Math.min(mRawXs.length, mRawYs.length), mRawIds.length);
        for (int i = 1; i < n; i++) {
            if (mRawIds[i] != mRawIds[i - 1]) continue;
            mLinePaint.setColor(colorForPointer(mRawIds[i]));
            canvas.drawLine(mRawXs[i - 1], mRawYs[i - 1], mRawXs[i], mRawYs[i], mLinePaint);
        }
    }

    private void drawRawPoints(final Canvas canvas) {
        if (mRawXs == null || mRawYs == null || mRawIds == null) return;
        final int n = Math.min(Math.min(mRawXs.length, mRawYs.length), mRawIds.length);
        for (int i = 0; i < n; i++) {
            mRawPaint.setColor(colorForPointer(mRawIds[i]));
            canvas.drawCircle(mRawXs[i], mRawYs[i], RAW_RADIUS_PX, mRawPaint);
        }
    }

    private void drawRunMarkers(final Canvas canvas) {
        if (mRawXs == null || mRawYs == null || mRawIds == null || mRawTimes == null) return;
        final int n = Math.min(Math.min(mRawXs.length, mRawYs.length),
                Math.min(mRawIds.length, mRawTimes.length));
        int start = 0;
        while (start < n) {
            int end = start + 1;
            while (end < n && mRawIds[end] == mRawIds[start]) {
                end++;
            }
            drawRunMarker(canvas, start, end);
            start = end;
        }
    }

    private void drawRunMarker(final Canvas canvas, final int start, final int end) {
        canvas.drawCircle(mRawXs[start], mRawYs[start], START_MARKER_RADIUS_PX, mStartPaint);
        canvas.drawRect(
                mRawXs[end - 1] - END_MARKER_RADIUS_PX,
                mRawYs[end - 1] - END_MARKER_RADIUS_PX,
                mRawXs[end - 1] + END_MARKER_RADIUS_PX,
                mRawYs[end - 1] + END_MARKER_RADIUS_PX,
                mEndPaint);
        final int pointCount = end - start;
        final int duration = mRawTimes[end - 1] - mRawTimes[start];
        if (pointCount <= TAP_MAX_POINTS && duration <= TAP_MAX_DURATION_MS) {
            final int mid = start + pointCount / 2;
            canvas.drawRect(
                    mRawXs[mid] - TAP_MARKER_RADIUS_PX,
                    mRawYs[mid] - TAP_MARKER_RADIUS_PX,
                    mRawXs[mid] + TAP_MARKER_RADIUS_PX,
                    mRawYs[mid] + TAP_MARKER_RADIUS_PX,
                    mTapPaint);
        }
    }

    private void drawSyntheticPoints(final Canvas canvas) {
        if (mSyntheticXs == null || mSyntheticYs == null) return;
        final int n = Math.min(mSyntheticXs.length, mSyntheticYs.length);
        for (int i = 0; i < n; i++) {
            final int x = mSyntheticXs[i];
            final int y = mSyntheticYs[i];
            canvas.drawLine(x - SYNTHETIC_CROSS_RADIUS_PX, y, x + SYNTHETIC_CROSS_RADIUS_PX,
                    y, mSyntheticPaint);
            canvas.drawLine(x, y - SYNTHETIC_CROSS_RADIUS_PX, x,
                    y + SYNTHETIC_CROSS_RADIUS_PX, mSyntheticPaint);
        }
    }

    private static int colorForPointer(final int pointerId) {
        return POINTER_COLORS[(pointerId & 0x7fffffff) % POINTER_COLORS.length];
    }

    private static int[] copyOfLength(final int[] src, final int length) {
        if (length <= 0) return null;
        final int[] copy = new int[length];
        System.arraycopy(src, 0, copy, 0, length);
        return copy;
    }
}
