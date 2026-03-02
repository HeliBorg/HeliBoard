package helium314.keyboard.latin.utils;

import java.util.Locale;

import helium314.keyboard.latin.common.StringUtils;

public enum RecapitalizeMode {
    ORIGINAL_MIXED_CASE {
        @Override
        public String apply(String text, int[] sortedSeparators, Locale locale) {
            return text;
        }
    },
    ALL_LOWER {
        @Override
        public String apply(String text, int[] sortedSeparators, Locale locale) {
            return text.toLowerCase(locale);
        }
    },
    FIRST_WORD_UPPER {
        @Override
        public String apply(String text, int[] sortedSeparators, Locale locale) {
            return StringUtils.capitalizeEachWord(text, sortedSeparators, locale);
        }
    },
    ALL_UPPER {
        @Override
        public String apply(String text, int[] sortedSeparators, Locale locale) {
            return text.toUpperCase(locale);
        }
    };

    private static RecapitalizeMode[] sCarousel = values();

    public static RecapitalizeMode of(final String string, final int[] sortedSeparators) {
        if (StringUtils.isIdenticalAfterUpcase(string)) {
            return ALL_UPPER;
        } else if (StringUtils.isIdenticalAfterDowncase(string)) {
            return ALL_LOWER;
        } else if (StringUtils.isIdenticalAfterCapitalizeEachWord(string, sortedSeparators)) {
            return FIRST_WORD_UPPER;
        } else {
            return ORIGINAL_MIXED_CASE;
        }
    }

    public static int count() {
        return sCarousel.length;
    }

    public final RecapitalizeMode rotate(boolean skipOriginalMixedCaseMode) {
        int position = ordinal() + 1;
        if (position == sCarousel.length) {
            position = skipOriginalMixedCaseMode ? 1 : 0;
        }
        return sCarousel[position];
    }

    public abstract String apply(String text, int[] sortedSeparators, Locale locale);
}
