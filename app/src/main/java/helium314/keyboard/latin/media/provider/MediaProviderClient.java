/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media.provider;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.RemoteException;
import android.util.Log;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

import helium314.keyboard.latin.media.IMediaProviderService;
import helium314.keyboard.latin.media.MediaPluginContract;

public final class MediaProviderClient {
    public interface CapabilitiesCallback {
        void onCapabilities(MediaProviderInfo provider);
        void onError(String message);
    }

    public interface SearchCallback {
        void onResults(List<MediaProviderItem> items, String nextPageToken);
        void onError(String message);
    }

    public interface ContentCallback {
        void onContent(MediaProviderItem item);
        void onError(String message);
    }

    public interface BrowseCallback {
        void onResults(List<MediaProviderItem> items, String nextPageToken);
        void onError(String message);
    }

    private final Context mContext;
    private final Handler mHandler = new Handler(Looper.getMainLooper());
    private final ExecutorService mExecutor = Executors.newSingleThreadExecutor();
    private final Set<ServiceConnection> mConnections =
            Collections.synchronizedSet(new HashSet<>());
    private volatile boolean mClosed;
    private static final long PLUGIN_TIMEOUT_MILLIS = 8000L;

    public MediaProviderClient(final Context context) {
        mContext = context.getApplicationContext();
    }

    public boolean hasProviders() {
        return !getProviders().isEmpty();
    }

    public List<MediaProviderInfo> getProviders() {
        final ArrayList<MediaProviderInfo> enabled = new ArrayList<>();
        for (final MediaProviderInfo provider : getDiscoveredProviders()) {
            if (MediaPluginApprovalStore.isEnabled(mContext, provider)) {
                enabled.add(provider);
            }
        }
        return enabled;
    }

    public List<MediaProviderInfo> getDiscoveredProviders() {
        final ArrayList<MediaProviderInfo> providers = new ArrayList<>();
        final PackageManager packageManager = mContext.getPackageManager();
        for (final ResolveInfo resolveInfo : queryProviders()) {
            if (resolveInfo.serviceInfo == null) {
                continue;
            }
            final String packageName = resolveInfo.serviceInfo.packageName;
            final String serviceName = resolveInfo.serviceInfo.name;
            final CharSequence label = resolveInfo.loadLabel(packageManager);
            providers.add(new MediaProviderInfo(packageName + "/" + serviceName,
                    label == null ? packageName : label.toString(), packageName, serviceName));
        }
        return providers;
    }

    public void close() {
        if (mClosed) {
            return;
        }
        mClosed = true;
        synchronized (mConnections) {
            for (final ServiceConnection connection : new ArrayList<>(mConnections)) {
                safeUnbind(connection);
            }
            mConnections.clear();
        }
        mExecutor.shutdownNow();
    }

    public void discoverCapabilities(final MediaProviderInfo provider,
            final CapabilitiesCallback callback) {
        bindProvider(provider, new BoundAction() {
            @Override
            public void run(final IMediaProviderService service, final ServiceConnection connection) {
                if (!tryExecute(() -> {
                    try {
                        final Bundle capabilities = service.discoverCapabilities();
                        final boolean supportsSearch = capabilities == null
                                || capabilities.getBoolean(
                                        MediaPluginContract.BUNDLE_SUPPORTS_SEARCH, true);
                        final boolean supportsBrowse = capabilities != null
                                && capabilities.getBoolean(
                                        MediaPluginContract.BUNDLE_SUPPORTS_BROWSE, false);
                        if (!isConnectionFinished(connection)) {
                            mHandler.post(() -> callback.onCapabilities(
                                    provider.withCapabilities(supportsSearch, supportsBrowse)));
                        }
                    } catch (RemoteException | RuntimeException e) {
                        Log.e(MediaPluginContract.LOG_TAG,
                                "media provider capabilities failed", e);
                        if (!isConnectionFinished(connection)) {
                            mHandler.post(() -> callback.onError("Media plugin unavailable"));
                        }
                    } finally {
                        safeUnbind(connection);
                    }
                }, "Media plugin unavailable", callback::onError)) {
                    safeUnbind(connection);
                }
            }

            @Override
            public void error(final String message) {
                callback.onError(message);
            }
        });
    }

    public void search(final MediaProviderInfo provider, final String query, final long maxBytes,
            final String pageToken,
            final SearchCallback callback) {
        bindProvider(provider, new BoundAction() {
            @Override
            public void run(final IMediaProviderService service, final ServiceConnection connection) {
                if (!tryExecute(() -> {
                    try {
                        final Bundle options = new Bundle();
                        options.putInt(MediaPluginContract.BUNDLE_LIMIT, 24);
                        options.putLong(MediaPluginContract.BUNDLE_MAX_BYTES, maxBytes);
                        if (pageToken != null && !pageToken.isEmpty()) {
                            options.putString(MediaPluginContract.BUNDLE_PAGE_TOKEN, pageToken);
                        }
                        final Bundle result = service.search(query, options);
                        final List<MediaProviderItem> items = parseItems(result);
                        final String nextPageToken = result == null ? null
                                : result.getString(MediaPluginContract.BUNDLE_NEXT_PAGE_TOKEN);
                        if (!isConnectionFinished(connection)) {
                            mHandler.post(() -> callback.onResults(items, nextPageToken));
                        }
                    } catch (RemoteException | RuntimeException e) {
                        Log.e(MediaPluginContract.LOG_TAG, "media provider search failed", e);
                        if (!isConnectionFinished(connection)) {
                            mHandler.post(() -> callback.onError("Search failed"));
                        }
                    } finally {
                        safeUnbind(connection);
                    }
                }, "Search failed", callback::onError)) {
                    safeUnbind(connection);
                }
            }

            @Override
            public void error(final String message) {
                callback.onError(message);
            }
        });
    }

    public void getContent(final MediaProviderInfo provider, final String itemId, final long maxBytes,
            final ContentCallback callback) {
        bindProvider(provider, new BoundAction() {
            @Override
            public void run(final IMediaProviderService service, final ServiceConnection connection) {
                if (!tryExecute(() -> {
                    try {
                        final Bundle options = new Bundle();
                        options.putLong(MediaPluginContract.BUNDLE_MAX_BYTES, maxBytes);
                        final MediaProviderItem item =
                                MediaProviderItem.fromBundle(service.getContent(itemId, options));
                        if (item == null || item.contentUri == null) {
                            if (!isConnectionFinished(connection)) {
                                mHandler.post(() -> callback.onError("No media returned"));
                            }
                        } else {
                            if (!isConnectionFinished(connection)) {
                                mHandler.post(() -> callback.onContent(item));
                            }
                        }
                    } catch (RemoteException | RuntimeException e) {
                        Log.e(MediaPluginContract.LOG_TAG, "media provider content failed", e);
                        if (!isConnectionFinished(connection)) {
                            mHandler.post(() -> callback.onError("Download failed"));
                        }
                    } finally {
                        safeUnbind(connection);
                    }
                }, "Download failed", callback::onError)) {
                    safeUnbind(connection);
                }
            }

            @Override
            public void error(final String message) {
                callback.onError(message);
            }
        });
    }

    public void browse(final MediaProviderInfo provider, @Nullable final String parentId,
            final long maxBytes, final String pageToken, final BrowseCallback callback) {
        bindProvider(provider, new BoundAction() {
            @Override
            public void run(final IMediaProviderService service, final ServiceConnection connection) {
                if (!tryExecute(() -> {
                    try {
                        final Bundle options = new Bundle();
                        options.putInt(MediaPluginContract.BUNDLE_LIMIT, 100);
                        options.putLong(MediaPluginContract.BUNDLE_MAX_BYTES, maxBytes);
                        if (pageToken != null && !pageToken.isEmpty()) {
                            options.putString(MediaPluginContract.BUNDLE_PAGE_TOKEN, pageToken);
                        }
                        final Bundle result = service.browse(parentId, options);
                        final List<MediaProviderItem> items = parseItems(result);
                        final String nextPageToken = result == null ? null
                                : result.getString(MediaPluginContract.BUNDLE_NEXT_PAGE_TOKEN);
                        if (!isConnectionFinished(connection)) {
                            mHandler.post(() -> callback.onResults(items, nextPageToken));
                        }
                    } catch (RemoteException | RuntimeException e) {
                        Log.e(MediaPluginContract.LOG_TAG, "media provider browse failed", e);
                        if (!isConnectionFinished(connection)) {
                            mHandler.post(() -> callback.onError("Browse failed"));
                        }
                    } finally {
                        safeUnbind(connection);
                    }
                }, "Browse failed", callback::onError)) {
                    safeUnbind(connection);
                }
            }

            @Override
            public void error(final String message) {
                callback.onError(message);
            }
        });
    }

    private List<ResolveInfo> queryProviders() {
        final Intent intent = new Intent(MediaPluginContract.ACTION_MEDIA_PROVIDER);
        return mContext.getPackageManager().queryIntentServices(intent, PackageManager.MATCH_DEFAULT_ONLY);
    }

    private void bindProvider(final MediaProviderInfo provider, final BoundAction action) {
        if (mClosed) {
            action.error("Media plugin unavailable");
            return;
        }
        final MediaProviderInfo resolvedProvider;
        if (provider == null) {
            final List<MediaProviderInfo> providers = getProviders();
            if (providers.isEmpty()) {
                action.error("No media plugin installed");
                return;
            }
            resolvedProvider = providers.get(0);
        } else {
            resolvedProvider = provider;
        }
        if (resolvedProvider == null) {
            action.error("No media plugin installed");
            return;
        }
        if (!MediaPluginApprovalStore.isEnabled(mContext, resolvedProvider)) {
            action.error("Media plugin disabled");
            return;
        }
        final Intent intent = new Intent(MediaPluginContract.ACTION_MEDIA_PROVIDER);
        intent.setClassName(resolvedProvider.packageName, resolvedProvider.serviceName);
        final AtomicBoolean finished = new AtomicBoolean(false);
        final ServiceConnection connection = new ServiceConnection() {
            @Override
            public void onServiceConnected(final ComponentName name, final IBinder binder) {
                if (mClosed || finished.get()) {
                    safeUnbind(this);
                    return;
                }
                action.run(IMediaProviderService.Stub.asInterface(binder),
                        new TimeoutServiceConnection(this, finished));
            }

            @Override
            public void onServiceDisconnected(final ComponentName name) {
            }
        };
        mConnections.add(connection);
        mHandler.postDelayed(() -> {
            if (finished.compareAndSet(false, true)) {
                safeUnbind(connection);
                action.error("Media plugin timed out");
            }
        }, PLUGIN_TIMEOUT_MILLIS);
        if (!mContext.bindService(intent, connection, Context.BIND_AUTO_CREATE)) {
            finished.set(true);
            safeUnbind(connection);
            action.error("Could not bind media plugin");
        }
    }

    private List<MediaProviderItem> parseItems(final Bundle result) {
        final ArrayList<MediaProviderItem> items = new ArrayList<>();
        if (result == null) {
            return items;
        }
        final ArrayList<Bundle> bundles =
                result.getParcelableArrayList(MediaPluginContract.BUNDLE_ITEMS);
        if (bundles == null) {
            return items;
        }
        for (final Bundle bundle : bundles) {
            final MediaProviderItem item = MediaProviderItem.fromBundle(bundle);
            if (item != null) {
                items.add(item);
            }
        }
        return items;
    }

    private void safeUnbind(final ServiceConnection connection) {
        if (connection instanceof TimeoutServiceConnection) {
            ((TimeoutServiceConnection) connection).finish();
            return;
        }
        mConnections.remove(connection);
        try {
            mContext.unbindService(connection);
        } catch (IllegalArgumentException ignored) {
        }
    }

    private boolean tryExecute(final Runnable runnable, final String errorMessage,
            final ErrorCallback callback) {
        if (mClosed) {
            callback.onError("Media plugin unavailable");
            return false;
        }
        try {
            mExecutor.execute(runnable);
            return true;
        } catch (RejectedExecutionException e) {
            callback.onError(errorMessage);
            return false;
        }
    }

    private boolean isConnectionFinished(final ServiceConnection connection) {
        return connection instanceof TimeoutServiceConnection
                && ((TimeoutServiceConnection) connection).isFinished();
    }

    private interface BoundAction {
        void run(IMediaProviderService service, ServiceConnection connection);
        void error(String message);
    }

    private interface ErrorCallback {
        void onError(String message);
    }

    private final class TimeoutServiceConnection implements ServiceConnection {
        private final ServiceConnection mDelegate;
        private final AtomicBoolean mFinished;

        TimeoutServiceConnection(final ServiceConnection delegate, final AtomicBoolean finished) {
            mDelegate = delegate;
            mFinished = finished;
        }

        @Override
        public void onServiceConnected(final ComponentName name, final IBinder service) {
        }

        @Override
        public void onServiceDisconnected(final ComponentName name) {
        }

        void finish() {
            mFinished.set(true);
            safeUnbind(mDelegate);
        }

        boolean isFinished() {
            return mFinished.get();
        }
    }
}
