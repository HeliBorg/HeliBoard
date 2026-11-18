/*
 * Copyright (C) 2013 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */

package helium314.keyboard.latin.common;

import android.content.Context;
import android.net.Uri;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.CountDownLatch;

import helium314.keyboard.latin.utils.ExecutorUtils;

/**
 * A simple class to help with removing directories recursively.
 */
public class FileUtils {

    public static boolean deleteRecursively(final File path) {
        if (path.isDirectory()) {
            final File[] files = path.listFiles();
            if (files != null) {
                for (final File child : files) {
                    deleteRecursively(child);
                }
            }
        }
        return path.delete();
    }

    /**
     *  copy data to file on different thread to avoid NetworkOnMainThreadException
     *  still effectively blocking, as we only use small files which are mostly stored locally
     */
    public static void copyContentUriToNewFile(final Uri uri, final Context context, final File outfile) throws IOException {
        final boolean[] allOk = new boolean[] { true };
        final CountDownLatch wait = new CountDownLatch(1);
        ExecutorUtils.getBackgroundExecutor(ExecutorUtils.KEYBOARD).execute(() -> {
            try {
                copyStreamToNewFile(context.getContentResolver().openInputStream(uri), outfile);
            } catch (IOException e) {
                allOk[0] = false;
            } finally {
                wait.countDown();
            }
        });
        try {
            wait.await();
        } catch (InterruptedException e) {
            allOk[0] = false;
        }
        if (!allOk[0])
            throw new IOException("could not copy from uri");
    }

    public static void copyStreamToNewFile(final InputStream in, final File outfile) throws IOException {
        File parentFile = outfile.getParentFile();
        if (parentFile == null || (!parentFile.exists() && !parentFile.mkdirs())) {
            throw new IOException("could not create parent folder");
        }
        FileOutputStream out = new FileOutputStream(outfile);
        copyStreamToOtherStream(in, out);
        out.close();
    }

    public static void copyStreamToOtherStream(final InputStream in, final OutputStream out) throws IOException {
        byte[] buf = new byte[1024];
        int len;
        while ((len = in.read(buf)) > 0) {
            out.write(buf, 0, len);
        }
        out.flush();
    }

}
