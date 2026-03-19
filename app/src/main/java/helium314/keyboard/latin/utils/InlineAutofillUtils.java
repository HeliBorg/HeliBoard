/*
 * Copyright (C) 2019 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0
 */

package helium314.keyboard.latin.utils;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.drawable.Icon;
import android.os.Build;
import android.os.Bundle;
import android.util.AttributeSet;
import android.util.Size;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.view.inputmethod.InlineSuggestion;
import android.view.inputmethod.InlineSuggestionsRequest;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.inline.InlineContentView;
import android.widget.inline.InlinePresentationSpec;

import androidx.annotation.AttrRes;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.autofill.inline.UiVersions;
import androidx.autofill.inline.UiVersions.StylesBuilder;
import androidx.autofill.inline.common.ImageViewStyle;
import androidx.autofill.inline.common.TextViewStyle;
import androidx.autofill.inline.common.ViewStyle;
import androidx.autofill.inline.v1.InlineSuggestionUi;
import androidx.autofill.inline.v1.InlineSuggestionUi.Style;

import java.util.ArrayList;
import java.util.List;

import helium314.keyboard.latin.R;
import helium314.keyboard.latin.common.ColorType;
import helium314.keyboard.latin.common.Colors;
import helium314.keyboard.latin.settings.Settings;

@SuppressLint("RestrictedApi")
@RequiresApi(api = Build.VERSION_CODES.R)
public class InlineAutofillUtils {

    public static InlineSuggestionsRequest createInlineSuggestionRequest(final Context context) {
        final Colors colors = Settings.getValues().mColors;
        final int chipBgDrawableId = androidx.autofill.R.drawable.autofill_inline_suggestion_chip_background;
        final int chipBgColor = colors.get(ColorType.AUTOFILL_BACKGROUND_CHIP);
        final int chipTextColor = colors.get(ColorType.KEY_TEXT);
        final int chipTextHintColor = colors.get(ColorType.KEY_HINT_TEXT);

        StylesBuilder stylesBuilder = UiVersions.newStylesBuilder();
        Style style = InlineSuggestionUi.newStyleBuilder()
                .setSingleIconChipStyle(
                        new ViewStyle.Builder()
                                .setBackground(Icon.createWithResource(context, chipBgDrawableId).setTint(chipBgColor))
                                .setPadding(0, 0, 0, 0)
                                .build())
                .setChipStyle(
                        new ViewStyle.Builder()
                                .setBackground(Icon.createWithResource(context, chipBgDrawableId).setTint(chipBgColor))
                                .build())
                .setStartIconStyle(new ImageViewStyle.Builder().setLayoutMargin(0, 0, 0, 0).build())
                .setTitleStyle(
                        new TextViewStyle.Builder()
                                .setTextColor(chipTextColor)
                                .setTextSize(12)
                                .build())
                .setSubtitleStyle(
                        new TextViewStyle.Builder()
                                .setTextColor(chipTextHintColor)
                                .setTextSize(10)
                                .build())
                .setEndIconStyle(new ImageViewStyle.Builder().setLayoutMargin(0, 0, 0, 0).build())
                .build();
        stylesBuilder.addStyle(style);
        Bundle stylesBundle = stylesBuilder.build();

        final int height = context.getResources().getDimensionPixelSize(R.dimen.config_suggestions_strip_height);
        final Size min = new Size(100, height);
        final Size max = new Size(740, height);

        // Three InlinePresentationSpec are required for some password managers
        final ArrayList<InlinePresentationSpec> presentationSpecs = new ArrayList<>();
        presentationSpecs.add(new InlinePresentationSpec.Builder(min, max).setStyle(stylesBundle).build());
        presentationSpecs.add(new InlinePresentationSpec.Builder(min, max).setStyle(stylesBundle).build());
        presentationSpecs.add(new InlinePresentationSpec.Builder(min, max).setStyle(stylesBundle).build());

        return new InlineSuggestionsRequest.Builder(presentationSpecs)
                .setMaxSuggestionCount(6)
                .build();
    }

    public static InlineContentClipView createView(final List<InlineSuggestion> inlineSuggestions,
                                                   final Context context) {
        LinearLayout container = new LinearLayout(context);
        for (InlineSuggestion inlineSuggestion : inlineSuggestions) {
            inlineSuggestion.inflate(context, new Size(ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT), context.getMainExecutor(), (view) -> {
                if (view != null)
                    container.addView(view);
            });
        }

        HorizontalScrollView inlineSuggestionView = new HorizontalScrollView(context);
        inlineSuggestionView.setHorizontalScrollBarEnabled(false);
        inlineSuggestionView.setOverScrollMode(View.OVER_SCROLL_NEVER);
        inlineSuggestionView.addView(container);

        InlineContentClipView scrollableSuggestionsClip = new InlineContentClipView(context);
        scrollableSuggestionsClip.addView(inlineSuggestionView);
        return scrollableSuggestionsClip;
    }

    /**
     * This class is a container for showing {@link InlineContentView}s for cases
     * where you want to ensure they appear only in a given area in your app. An
     * example is having a scrollable list of items. Note that without this container
     * the InlineContentViews' surfaces would cover parts of your app as these surfaces
     * are owned by another process and always appearing on top of your app.
     */
    public static class InlineContentClipView extends FrameLayout {
        @NonNull
        private final Rect mParentBounds = new Rect();
        @NonNull
        private final Rect mContentBounds = new Rect();
        public InlineContentClipView(@NonNull Context context) {
            this(context, /*attrs*/ null, /*defStyleAttr*/ 0);
        }
        public InlineContentClipView(@NonNull Context context, @Nullable AttributeSet attrs,
                                     @AttrRes int defStyleAttr) {
            super(context, attrs, defStyleAttr);
            SurfaceView mBackgroundView = new SurfaceView(context);
            mBackgroundView.setZOrderOnTop(true);
            mBackgroundView.getHolder().setFormat(PixelFormat.TRANSPARENT);
            addView(mBackgroundView);
        }

        @Override
        protected void onLayout(boolean changed, int left, int top, int right, int bottom) {
            super.onLayout(changed, left, top, right, bottom);
            if (changed) {
                clipDescendantInlineContentViews();
            }
        }

        private void clipDescendantInlineContentViews() {
            mParentBounds.right = getWidth();
            mParentBounds.bottom = getHeight();
            Log.d(InlineContentClipView.class.getSimpleName(), "Clipping to " + mParentBounds);
            clipDescendantInlineContentViews(this);
        }
        private void clipDescendantInlineContentViews(@Nullable View root) {
            if (root == null) return;
            if (root instanceof InlineContentView inlineContentView) {
                mContentBounds.set(mParentBounds);
                offsetRectIntoDescendantCoords(inlineContentView, mContentBounds);
                inlineContentView.setClipBounds(mContentBounds);
                return;
            }
            if (root instanceof ViewGroup rootGroup) {
                final int childCount = rootGroup.getChildCount();
                for (int i = 0; i < childCount; i++) {
                    final View child = rootGroup.getChildAt(i);
                    clipDescendantInlineContentViews(child);
                }
            }
        }
    }
}
