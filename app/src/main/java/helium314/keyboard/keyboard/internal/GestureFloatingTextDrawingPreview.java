/*
 * Copyright (C) 2012 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.keyboard.internal;

import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Paint.Align;
import android.graphics.Rect;
import android.graphics.RectF;
import android.text.TextUtils;

import androidx.annotation.NonNull;

import helium314.keyboard.keyboard.KeyboardTypeface;
import helium314.keyboard.keyboard.PointerTracker;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.SuggestedWords;
import helium314.keyboard.latin.common.ColorType;
import helium314.keyboard.latin.common.Colors;
import helium314.keyboard.latin.common.CoordinateUtils;
import helium314.keyboard.latin.settings.Settings;

/**
 * The class for single gesture preview text. The class for multiple gesture preview text will be
 * derived from it.
 */
public class GestureFloatingTextDrawingPreview extends AbstractDrawingPreview {
    protected static final class GesturePreviewTextParams {
        public final boolean mGesturePreviewDynamic;
        public final int mGesturePreviewTextOffset;
        public final int mGesturePreviewTextHeight;
        public final float mGesturePreviewHorizontalPadding;
        public final float mGesturePreviewVerticalPadding;
        public final float mGesturePreviewRoundRadius;
        public final int mDisplayWidth;

        private final int mGesturePreviewTextSize;
        private final int mGesturePreviewTextColor;
        private final int mGesturePreviewColor;
        private final Paint mPaint = new Paint();

        private static final char[] TEXT_HEIGHT_REFERENCE_CHAR = { 'M' };

        public GesturePreviewTextParams(final TypedArray mainKeyboardViewAttr) {
            final Colors colors = Settings.getValues().mColors;
            mGesturePreviewDynamic = Settings.getValues().mGestureFloatingPreviewDynamicEnabled;
            mGesturePreviewTextSize = mainKeyboardViewAttr.getDimensionPixelSize(
                    R.styleable.MainKeyboardView_gestureFloatingPreviewTextSize, 0);
            mGesturePreviewTextColor = colors.get(ColorType.KEY_TEXT);
            mGesturePreviewTextOffset = mainKeyboardViewAttr.getDimensionPixelOffset(
                    R.styleable.MainKeyboardView_gestureFloatingPreviewTextOffset, 0);
            mGesturePreviewColor = colors.get(ColorType.GESTURE_PREVIEW);
            mGesturePreviewHorizontalPadding = mainKeyboardViewAttr.getDimension(
                    R.styleable.MainKeyboardView_gestureFloatingPreviewHorizontalPadding, 0.0f);
            mGesturePreviewVerticalPadding = mainKeyboardViewAttr.getDimension(
                    R.styleable.MainKeyboardView_gestureFloatingPreviewVerticalPadding, 0.0f);
            mGesturePreviewRoundRadius = mainKeyboardViewAttr.getDimension(
                    R.styleable.MainKeyboardView_gestureFloatingPreviewRoundRadius, 0.0f);
            mDisplayWidth = mainKeyboardViewAttr.getResources().getDisplayMetrics().widthPixels;

            final Paint textPaint = getTextPaint();
            final Rect textRect = new Rect();
            textPaint.getTextBounds(TEXT_HEIGHT_REFERENCE_CHAR, 0, 1, textRect);
            mGesturePreviewTextHeight = textRect.height();
        }

        public Paint getTextPaint() {
            mPaint.setAntiAlias(true);
            mPaint.setTextAlign(Align.CENTER);
            mPaint.setTextSize(mGesturePreviewTextSize);
            mPaint.setTypeface(KeyboardTypeface.customTypeface());
            mPaint.setColor(mGesturePreviewTextColor);
            return mPaint;
        }

        public Paint getBackgroundPaint() {
            mPaint.setColor(mGesturePreviewColor);
            return mPaint;
        }
    }

    private final GesturePreviewTextParams mParams;
    private final RectF mGesturePreviewRectangle = new RectF();
    private int mPreviewTextX;
    private int mPreviewTextY;
    private SuggestedWords mSuggestedWords = SuggestedWords.getEmptyInstance();
    private final int[] mLastPointerCoords = CoordinateUtils.newInstance();

    public GestureFloatingTextDrawingPreview(final TypedArray mainKeyboardViewAttr) {
        mParams = new GesturePreviewTextParams(mainKeyboardViewAttr);
    }

    @Override
    public void onDeallocateMemory() {
        // Nothing to do here.
    }

    public void dismissGestureFloatingPreviewText() {
        setSuggestedWords(SuggestedWords.getEmptyInstance());
    }

    public void setSuggestedWords(@NonNull final SuggestedWords suggestedWords) {
        if (!isPreviewEnabled()) {
            return;
        }
        mSuggestedWords = suggestedWords;
        updatePreviewPosition();
    }

    @Override
    public void setPreviewPosition(@NonNull final PointerTracker tracker) {
        if (!isPreviewEnabled()) {
            return;
        }
        tracker.getLastCoordinates(mLastPointerCoords);
        updatePreviewPosition();
    }

    /**
     * Draws gesture preview text
     * @param canvas The canvas where preview text is drawn.
     */
    @Override
    public void drawPreview(@NonNull final Canvas canvas) {
        if (!isPreviewEnabled() || mSuggestedWords.isEmpty()
                || TextUtils.isEmpty(mSuggestedWords.getWord(0))) {
            return;
        }
        final float round = mParams.mGesturePreviewRoundRadius;
        canvas.drawRoundRect(
                mGesturePreviewRectangle, round, round, mParams.getBackgroundPaint());
        final String text = mSuggestedWords.getWord(0);
        canvas.drawText(text, mPreviewTextX, mPreviewTextY, mParams.getTextPaint());
    }

    /**
     * Updates gesture preview text position based on mLastPointerCoords.
     */
    protected void updatePreviewPosition() {
        if (mSuggestedWords.isEmpty() || TextUtils.isEmpty(mSuggestedWords.getWord(0))) {
            invalidateDrawingView();
            return;
        }
        final String text = mSuggestedWords.getWord(0);

        final int textHeight = mParams.mGesturePreviewTextHeight;
        final float textWidth = mParams.getTextPaint().measureText(text);
        final float hPad = mParams.mGesturePreviewHorizontalPadding;
        final float vPad = mParams.mGesturePreviewVerticalPadding;
        final float rectWidth = textWidth + hPad * 2.0f;
        final float rectHeight = textHeight + vPad * 2.0f;

        final float rectX = mParams.mGesturePreviewDynamic ? Math.min(
                Math.max(CoordinateUtils.x(mLastPointerCoords) - rectWidth / 2.0f, 0.0f),
                mParams.mDisplayWidth - rectWidth)
            : (mParams.mDisplayWidth - rectWidth) / 2.0f;
        final float rectY = mParams.mGesturePreviewDynamic ? CoordinateUtils.y(mLastPointerCoords)
                - mParams.mGesturePreviewTextOffset - rectHeight
            : -mParams.mGesturePreviewTextOffset - rectHeight;
        mGesturePreviewRectangle.set(rectX, rectY, rectX + rectWidth, rectY + rectHeight);

        mPreviewTextX = (int)(rectX + hPad + textWidth / 2.0f);
        mPreviewTextY = (int)(rectY + vPad) + textHeight;
        // TODO: Should narrow the invalidate region.
        invalidateDrawingView();
    }
}
