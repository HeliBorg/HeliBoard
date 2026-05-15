/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media;

import androidx.annotation.Nullable;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public final class MediaShareTargetCompatibility {
    private static final Set<String> PRIVATE_URI_REJECTED_PACKAGES = new HashSet<>(Arrays.asList(
            "com.android.messaging"
    ));

    private MediaShareTargetCompatibility() {
    }

    public static boolean isPrivateUriRejectedTarget(@Nullable final String packageName) {
        return packageName != null && PRIVATE_URI_REJECTED_PACKAGES.contains(packageName);
    }
}
