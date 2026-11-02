/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.utils;

import android.os.Build;
import android.text.Html;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.SpannedString;
import android.text.TextUtils;
import android.text.style.SuggestionSpan;
import android.text.style.URLSpan;

import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class SpannableStringUtils {
    /**
     * Copies the spans from the region <code>start...end</code> in
     * <code>source</code> to the region
     * <code>destoff...destoff+end-start</code> in <code>dest</code>.
     * Spans in <code>source</code> that begin before <code>start</code>
     * or end after <code>end</code> but overlap this range are trimmed
     * as if they began at <code>start</code> or ended at <code>end</code>.
     * Only SuggestionSpans that don't have the SPAN_PARAGRAPH span are copied.
     * <p>
     * This code is almost entirely taken from {@link TextUtils#copySpansFrom}, except for the
     * kind of span that is copied.
     *
     * @throws IndexOutOfBoundsException if any of the copied spans
     * are out of range in <code>dest</code>.
     */
    public static void copyNonParagraphSuggestionSpansFrom(Spanned source, int start, int end,
            Spannable dest, int destoff) {
        Object[] spans = source.getSpans(start, end, SuggestionSpan.class);

        for (Object span : spans) {
            int fl = source.getSpanFlags(span);
            // We don't care about the PARAGRAPH flag in LatinIME code. However, if this flag
            // is set, Spannable#setSpan will throw an exception unless the span is on the edge
            // of a word. But the spans have been split into two by the getText{Before,After}Cursor
            // methods, so after concatenation they may end in the middle of a word.
            // Since we don't use them, we can just remove them and avoid crashing.
            fl &= ~Spanned.SPAN_PARAGRAPH;

            int st = source.getSpanStart(span);
            int en = source.getSpanEnd(span);

            if (st < start)
                st = start;
            if (en > end)
                en = end;

            dest.setSpan(span, st - start + destoff, en - start + destoff, fl);
        }
    }

    /**
     * Returns a CharSequence concatenating the specified CharSequences, retaining their
     * SuggestionSpans that don't have the PARAGRAPH flag, but not other spans.
     * <p>
     * This code is almost entirely taken from {@link TextUtils#concat(CharSequence...)}, except
     * it calls copyNonParagraphSuggestionSpansFrom instead of {@link TextUtils#copySpansFrom}.
     */
    // We don't use TextUtils#concat because it copies all spans without respect to their
    // nature. If the text includes a PARAGRAPH span and it has been split, then
    // TextUtils#concat will crash when it tries to concat both sides of it.
    public static CharSequence concatWithNonParagraphSuggestionSpansOnly(CharSequence... text) {
        if (text.length == 0) {
            return "";
        }

        if (text.length == 1) {
            return text[0];
        }

        boolean spanned = false;
        for (CharSequence value : text) {
            if (value instanceof Spanned) {
                spanned = true;
                break;
            }
        }

        StringBuilder sb = new StringBuilder();
        for (CharSequence sequence : text) {
            sb.append(sequence);
        }

        if (!spanned) {
            return sb.toString();
        }

        SpannableString ss = new SpannableString(sb);
        int off = 0;
        for (CharSequence charSequence : text) {
            int len = charSequence.length();

            if (charSequence instanceof Spanned) {
                copyNonParagraphSuggestionSpansFrom((Spanned) charSequence, 0, len, ss, off);
            }

            off += len;
        }

        return new SpannedString(ss);
    }

    @SuppressWarnings("deprecation")
    public static Spanned fromHtml(final String text) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N)
            return Html.fromHtml(text, Html.FROM_HTML_MODE_LEGACY);
        return Html.fromHtml(text);
    }

    public static boolean hasUrlSpans(final CharSequence text,
            final int startIndex, final int endIndex) {
        if (!(text instanceof Spanned spanned)) {
            return false; // Not spanned, so no link
        }
        // getSpans(x, y) does not return spans that start on x or end on y. x-1, y+1 does the
        // trick, and works in all cases even if startIndex <= 0 or endIndex >= text.length().
        final URLSpan[] spans = spanned.getSpans(startIndex - 1, endIndex + 1, URLSpan.class);
        return null != spans && spans.length > 0;
    }

    /**
     * Splits the given {@code charSequence} with at occurrences of the given {@code regex}.
     * <p>
     * This is equivalent to
     * {@code charSequence.toString().split(regex, preserveTrailingEmptySegments ? -1 : 0)}
     * except that the spans are preserved in the result array.
     * </p>
     * @param charSequence the character sequence to be split.
     * @param regex the regex pattern to be used as the separator.
     * @param preserveTrailingEmptySegments {@code true} to preserve the trailing empty
     * segments. Otherwise, trailing empty segments will be removed before being returned.
     * @return the array which contains the result. All the spans in the <code>charSequence</code>
     * is preserved.
     */
    public static CharSequence[] split(final CharSequence charSequence, final String regex,
            final boolean preserveTrailingEmptySegments) {
        // A short-cut for non-spanned strings.
        if (!(charSequence instanceof Spanned)) {
            // -1 means that trailing empty segments will be preserved.
            return charSequence.toString().split(regex, preserveTrailingEmptySegments ? -1 : 0);
        }

        // Hereafter, emulate String.split for CharSequence.
        final ArrayList<CharSequence> sequences = new ArrayList<>();
        final Matcher matcher = Pattern.compile(regex).matcher(charSequence);
        int nextStart = 0;
        boolean matched = false;
        while (matcher.find()) {
            sequences.add(charSequence.subSequence(nextStart, matcher.start()));
            nextStart = matcher.end();
            matched = true;
        }
        if (!matched) {
            // never matched. preserveTrailingEmptySegments is ignored in this case.
            return new CharSequence[] { charSequence };
        }
        sequences.add(charSequence.subSequence(nextStart, charSequence.length()));
        if (!preserveTrailingEmptySegments) {
            for (int i = sequences.size() - 1; i >= 0; --i) {
                if (!TextUtils.isEmpty(sequences.get(i))) {
                    break;
                }
                sequences.remove(i);
            }
        }
        return sequences.toArray(new CharSequence[0]);
    }
}
