/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.utils;

import android.util.JsonReader;

import java.io.Closeable;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

// todo: remove around end of 2026 (used only for upgrading old recent emoji pref)
public final class JsonUtils {
    private static final String TAG = JsonUtils.class.getSimpleName();

    private static final String INTEGER_CLASS_NAME = Integer.class.getSimpleName();
    private static final String STRING_CLASS_NAME = String.class.getSimpleName();

    private static final String EMPTY_STRING = "";

    public static List<Object> jsonStrToList(final String s) {
        final ArrayList<Object> list = new ArrayList<>();
        final JsonReader reader = new JsonReader(new StringReader(s));
        try {
            reader.beginArray();
            while (reader.hasNext()) {
                reader.beginObject();
                while (reader.hasNext()) {
                    final String name = reader.nextName();
                    if (name.equals(INTEGER_CLASS_NAME)) {
                        list.add(reader.nextInt());
                    } else if (name.equals(STRING_CLASS_NAME)) {
                        list.add(reader.nextString());
                    } else {
                        Log.w(TAG, "Invalid name: " + name);
                        reader.skipValue();
                    }
                }
                reader.endObject();
            }
            reader.endArray();
            return list;
        } catch (final IOException ignored) {
        } finally {
            close(reader);
        }
        return Collections.emptyList();
    }

    private static void close(final Closeable closeable) {
        try {
            if (closeable != null) {
                closeable.close();
            }
        } catch (final IOException e) {
            // Ignore
        }
    }
}
