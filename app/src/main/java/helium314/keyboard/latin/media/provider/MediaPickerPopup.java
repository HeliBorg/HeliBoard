/*
 * Copyright (C) 2026
 * SPDX-License-Identifier: GPL-3.0-only
 */

package helium314.keyboard.latin.media.provider;

import android.graphics.ImageDecoder;
import android.graphics.drawable.AnimatedImageDrawable;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.content.SharedPreferences;
import android.util.LruCache;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import helium314.keyboard.keyboard.KeyboardSwitcher;
import helium314.keyboard.keyboard.internal.keyboard_parser.floris.KeyCode;
import helium314.keyboard.latin.LatinIME;
import helium314.keyboard.latin.R;
import helium314.keyboard.latin.common.ColorType;
import helium314.keyboard.latin.common.Colors;
import helium314.keyboard.latin.common.Constants;
import helium314.keyboard.latin.settings.Settings;

public final class MediaPickerPopup {
    private static final int GRID_SPAN_COUNT = 3;
    private static final int SEARCH_PREFETCH_THRESHOLD_ITEMS = 24;
    private static final int SEARCH_PREFETCH_TARGET_ITEMS = 72;
    private static final String PREFS_NAME = "media_provider_prefs";
    private static final String PREF_SELECTED_PROVIDER = "selected_provider";
    private static final String PREF_SELECTED_MODE = "selected_mode";
    private static final String MODE_SEARCH = "search";
    private static final String MODE_BROWSE = "browse";

    private final LatinIME mLatinIME;
    private final View mAnchor;
    private final long mMaxBytes;
    private final MediaProviderClient mClient;
    private final Colors mColors;
    private final ArrayList<MediaProviderItem> mItems = new ArrayList<>();
    private final ArrayList<MediaProviderInfo> mProviders = new ArrayList<>();
    private final ArrayList<ProviderChoice> mProviderChoices = new ArrayList<>();
    private final StringBuilder mQuery = new StringBuilder();
    private final Handler mHandler = new Handler(Looper.getMainLooper());
    private final ThreadPoolExecutor mPreviewExecutor =
            new ThreadPoolExecutor(2, 2, 0L, TimeUnit.MILLISECONDS,
                    new LinkedBlockingQueue<>());
    private final LruCache<String, Drawable.ConstantState> mPreviewCache =
            new LruCache<>(90);
    private boolean mCursorVisible = true;
    private PopupWindow mPopupWindow;
    private Button mProviderButton;
    private Button mSearchButton;
    private TextView mQueryView;
    private TextView mStatusView;
    private MediaAdapter mAdapter;
    private MediaProviderInfo mSelectedProvider;
    private String mSelectedMode = MODE_SEARCH;
    private String mCurrentQuery;
    private String mCurrentBrowseParent;
    private String mNextPageToken;
    private boolean mIsLoadingPage;
    private boolean mEndReached;
    private boolean mBrowseMode;
    private boolean mDismissed;
    private final ArrayList<BrowseLocation> mBrowseStack = new ArrayList<>();
    private final Runnable mCursorBlinkRunnable = new Runnable() {
        @Override
        public void run() {
            if (mPopupWindow == null || !mPopupWindow.isShowing()) {
                return;
            }
            mCursorVisible = !mCursorVisible;
            updateQueryView();
            mHandler.postDelayed(this, 500);
        }
    };

    public MediaPickerPopup(final LatinIME latinIME, final View anchor, final long maxBytes) {
        mLatinIME = latinIME;
        mAnchor = anchor;
        mMaxBytes = maxBytes;
        mClient = new MediaProviderClient(latinIME);
        mColors = Settings.getValues().mColors;
    }

    public void show() {
        loadProviders();
        KeyboardSwitcher.getInstance().setAlphabetKeyboard();
        mLatinIME.setActiveMediaPickerPopup(this);
        final float density = mAnchor.getResources().getDisplayMetrics().density;
        final LinearLayout root = new LinearLayout(mAnchor.getContext());
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(density, 8), dp(density, 8), dp(density, 8), dp(density, 8));
        final GradientDrawable background = new GradientDrawable();
        background.setColor(mColors.get(ColorType.MAIN_BACKGROUND));
        background.setCornerRadius(dp(density, 8));
        root.setBackground(background);

        final LinearLayout searchRow = new LinearLayout(mAnchor.getContext());
        searchRow.setOrientation(LinearLayout.HORIZONTAL);
        mQueryView = new TextView(mAnchor.getContext());
        mQueryView.setSingleLine(true);
        mQueryView.setTextSize(18);
        mQueryView.setGravity(Gravity.CENTER_VERTICAL);
        mQueryView.setTextColor(mColors.get(ColorType.KEY_TEXT));
        searchRow.addView(mQueryView, new LinearLayout.LayoutParams(0, dp(density, 48), 1));

        mSearchButton = new Button(mAnchor.getContext());
        mSearchButton.setText("Search");
        mSearchButton.setTextSize(12);
        mSearchButton.setAllCaps(false);
        searchRow.addView(mSearchButton, new LinearLayout.LayoutParams(dp(density, 96), dp(density, 48)));

        final Button cancelButton = new Button(mAnchor.getContext());
        cancelButton.setText("Cancel");
        cancelButton.setTextSize(12);
        cancelButton.setAllCaps(false);
        searchRow.addView(cancelButton, new LinearLayout.LayoutParams(dp(density, 84), dp(density, 48)));
        root.addView(searchRow);

        mProviderButton = new Button(mAnchor.getContext());
        mProviderButton.setTextSize(12);
        mProviderButton.setAllCaps(false);
        mProviderButton.setSingleLine(true);
        mProviderButton.setOnClickListener(view -> showProviderChooser());
        root.addView(mProviderButton, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(density, 40)));
        updateProviderButton();

        mStatusView = new TextView(mAnchor.getContext());
        mStatusView.setText("Search with installed media plugins");
        mStatusView.setTextColor(mColors.get(ColorType.KEY_TEXT));
        mStatusView.setGravity(Gravity.CENTER_VERTICAL);
        root.addView(mStatusView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(density, 32)));

        final RecyclerView recyclerView = new RecyclerView(mAnchor.getContext());
        final GridLayoutManager layoutManager =
                new GridLayoutManager(mAnchor.getContext(), GRID_SPAN_COUNT);
        recyclerView.setLayoutManager(layoutManager);
        mAdapter = new MediaAdapter();
        recyclerView.setAdapter(mAdapter);
        recyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override
            public void onScrolled(final RecyclerView recyclerView, final int dx, final int dy) {
                if (dy <= 0 || mItems.isEmpty()) {
                    return;
                }
                final int lastVisible = layoutManager.findLastVisibleItemPosition();
                if (lastVisible >= mItems.size() - SEARCH_PREFETCH_THRESHOLD_ITEMS) {
                    loadNextPage();
                }
            }
        });
        root.addView(recyclerView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        mSearchButton.setOnClickListener(view -> {
            if (mBrowseMode) {
                browseUp();
            } else {
                runSearch(mQuery.toString().trim());
            }
        });
        cancelButton.setOnClickListener(view -> dismiss());

        mPopupWindow = new PopupWindow(root, ViewGroup.LayoutParams.MATCH_PARENT,
                Math.min(dp(density, 520), Math.max(dp(density, 300), mAnchor.getHeight() / 2)),
                false);
        mPopupWindow.setClippingEnabled(false);
        mPopupWindow.setInputMethodMode(PopupWindow.INPUT_METHOD_NOT_NEEDED);
        mPopupWindow.setOnDismissListener(this::cleanupAfterDismiss);
        mPopupWindow.showAtLocation(mAnchor, Gravity.TOP, 0, 0);
        updateQueryView();
        mHandler.postDelayed(mCursorBlinkRunnable, 500);
        configureSelectedProvider();
    }

    private void loadProviders() {
        mProviders.clear();
        mProviders.addAll(mClient.getProviders());
        if (mProviders.isEmpty()) {
            mSelectedProvider = null;
            return;
        }
        final SharedPreferences prefs = mLatinIME.getSharedPreferences(PREFS_NAME, 0);
        final String selectedKey = prefs.getString(PREF_SELECTED_PROVIDER, null);
        mSelectedMode = prefs.getString(PREF_SELECTED_MODE, MODE_SEARCH);
        mSelectedProvider = mProviders.get(0);
        if (selectedKey != null) {
            for (final MediaProviderInfo provider : mProviders) {
                if (selectedKey.equals(provider.key)) {
                    mSelectedProvider = provider;
                    break;
                }
            }
        }
    }

    private void selectProviderChoice(final ProviderChoice choice) {
        mSelectedProvider = choice.provider;
        mSelectedMode = choice.mode;
        mLatinIME.getSharedPreferences(PREFS_NAME, 0).edit()
                .putString(PREF_SELECTED_PROVIDER, mSelectedProvider.key)
                .putString(PREF_SELECTED_MODE, mSelectedMode).apply();
        mItems.clear();
        mAdapter.notifyDataSetChanged();
        mCurrentQuery = null;
        mCurrentBrowseParent = null;
        mNextPageToken = null;
        mIsLoadingPage = false;
        mEndReached = false;
        mBrowseStack.clear();
        updateProviderButton();
        configureSelectedProvider();
    }

    private void showProviderChooser() {
        discoverProviderChoices(() -> {
            if (mProviderChoices.isEmpty()) {
                Toast.makeText(mLatinIME, "No media plugin enabled", Toast.LENGTH_SHORT).show();
                return;
            }
            if (mProviderChoices.size() == 1) {
                selectProviderChoice(mProviderChoices.get(0));
                return;
            }
            final float density = mAnchor.getResources().getDisplayMetrics().density;
            final LinearLayout list = new LinearLayout(mAnchor.getContext());
            list.setOrientation(LinearLayout.VERTICAL);
            list.setPadding(dp(density, 8), dp(density, 8), dp(density, 8), dp(density, 8));
            final GradientDrawable background = new GradientDrawable();
            background.setColor(mColors.get(ColorType.MAIN_BACKGROUND));
            background.setCornerRadius(dp(density, 8));
            list.setBackground(background);

            final PopupWindow chooser = new PopupWindow(list,
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT,
                    false);
            chooser.setClippingEnabled(false);
            for (final ProviderChoice choice : mProviderChoices) {
                final Button button = new Button(mAnchor.getContext());
                button.setText(choice.label());
                button.setTextSize(12);
                button.setAllCaps(false);
                button.setSingleLine(true);
                button.setOnClickListener(view -> {
                    chooser.dismiss();
                    selectProviderChoice(choice);
                });
                list.addView(button, new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT, dp(density, 44)));
            }
            chooser.showAtLocation(mAnchor, Gravity.TOP, 0, 0);
        });
    }

    private void discoverProviderChoices(final Runnable onComplete) {
        mProviderChoices.clear();
        if (mProviders.isEmpty()) {
            onComplete.run();
            return;
        }
        mStatusView.setText("Loading media providers...");
        final int[] remaining = new int[] { mProviders.size() };
        for (final MediaProviderInfo provider : mProviders) {
            mClient.discoverCapabilities(provider, new MediaProviderClient.CapabilitiesCallback() {
                @Override
                public void onCapabilities(final MediaProviderInfo providerWithCapabilities) {
                    addProviderChoices(providerWithCapabilities);
                    finishProviderDiscovery(remaining, onComplete);
                }

                @Override
                public void onError(final String message) {
                    finishProviderDiscovery(remaining, onComplete);
                }
            });
        }
    }

    private void finishProviderDiscovery(final int[] remaining, final Runnable onComplete) {
        remaining[0]--;
        if (remaining[0] == 0) {
            onComplete.run();
        }
    }

    private void addProviderChoices(final MediaProviderInfo provider) {
        if (provider.supportsSearch) {
            mProviderChoices.add(new ProviderChoice(provider, MODE_SEARCH));
        }
        if (provider.supportsBrowse) {
            mProviderChoices.add(new ProviderChoice(provider, MODE_BROWSE));
        }
    }

    private void updateProviderButton() {
        if (mProviderButton == null) {
            return;
        }
        if (mSelectedProvider == null) {
            mProviderButton.setText("No media plugin enabled");
            mProviderButton.setEnabled(true);
            return;
        }
        mProviderButton.setText("Provider: " + mSelectedProvider.label + " "
                + modeLabel(mSelectedMode));
        mProviderButton.setEnabled(true);
    }

    public boolean handleCodeInput(final int primaryCode) {
        if (mBrowseMode) {
            return true;
        }
        if (primaryCode == KeyCode.DELETE) {
            if (mQuery.length() > 0) {
                mQuery.deleteCharAt(mQuery.length() - 1);
                updateQueryView();
            }
            return true;
        }
        if (primaryCode == Constants.CODE_ENTER) {
            runSearch(mQuery.toString().trim());
            return true;
        }
        if (primaryCode == Constants.CODE_SPACE) {
            appendText(" ");
            return true;
        }
        if (primaryCode > 0 && !Character.isISOControl(primaryCode)) {
            appendText(new String(Character.toChars(primaryCode)));
            return true;
        }
        return false;
    }

    public boolean handleTextInput(@Nullable final String text) {
        if (mBrowseMode) {
            return true;
        }
        if (text == null || text.isEmpty()) {
            return false;
        }
        appendText(text);
        return true;
    }

    private void appendText(final String text) {
        mQuery.append(text);
        mCursorVisible = true;
        updateQueryView();
    }

    private void updateQueryView() {
        if (mQueryView != null) {
            if (mBrowseMode) {
                mQueryView.setText(currentBrowseTitle());
                return;
            }
            final String cursor = mCursorVisible ? "|" : " ";
            mQueryView.setText(mQuery.length() == 0 ? cursor + " Search media" : mQuery + cursor);
        }
    }

    private void configureSelectedProvider() {
        if (mSelectedProvider == null) {
            mBrowseMode = false;
            mItems.clear();
            if (mAdapter != null) {
                mAdapter.notifyDataSetChanged();
            }
            if (mStatusView != null) {
                if (mClient.getDiscoveredProviders().isEmpty()) {
                    mStatusView.setText("No media plugin installed");
                } else {
                    mStatusView.setText("Enable a media plugin in HeliBoard settings");
                }
            }
            updateProviderButton();
            updateModeControls();
            return;
        }
        mStatusView.setText("Loading " + mSelectedProvider.label + "...");
        mClient.discoverCapabilities(mSelectedProvider, new MediaProviderClient.CapabilitiesCallback() {
            @Override
            public void onCapabilities(final MediaProviderInfo provider) {
                if (!provider.key.equals(mSelectedProvider.key)) {
                    return;
                }
                mSelectedProvider = provider;
                if (MODE_BROWSE.equals(mSelectedMode) && !provider.supportsBrowse) {
                    mSelectedMode = provider.supportsSearch ? MODE_SEARCH : MODE_BROWSE;
                } else if (MODE_SEARCH.equals(mSelectedMode) && !provider.supportsSearch) {
                    mSelectedMode = provider.supportsBrowse ? MODE_BROWSE : MODE_SEARCH;
                }
                mBrowseMode = MODE_BROWSE.equals(mSelectedMode);
                mItems.clear();
                mAdapter.notifyDataSetChanged();
                mCurrentQuery = null;
                mCurrentBrowseParent = null;
                mNextPageToken = null;
                mIsLoadingPage = false;
                mEndReached = false;
                mBrowseStack.clear();
                updateProviderButton();
                updateModeControls();
                if (mBrowseMode) {
                    browseFolder(null, null);
                } else if (provider.supportsSearch) {
                    mStatusView.setText("Search with " + mSelectedProvider.label);
                } else {
                    mStatusView.setText("Media provider has no usable mode");
                }
            }

            @Override
            public void onError(final String message) {
                mStatusView.setText(message);
                Toast.makeText(mLatinIME, message, Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void updateModeControls() {
        if (mSearchButton == null) {
            return;
        }
        if (mBrowseMode) {
            mSearchButton.setText("Up");
            mSearchButton.setEnabled(!mBrowseStack.isEmpty());
        } else {
            mSearchButton.setText("Search");
            mSearchButton.setEnabled(true);
        }
        updateQueryView();
    }

    private void runSearch(final String query) {
        if (query.isEmpty()) {
            Toast.makeText(mLatinIME, "Enter a search term", Toast.LENGTH_SHORT).show();
            return;
        }
        mCurrentQuery = query;
        mNextPageToken = null;
        mIsLoadingPage = false;
        mEndReached = false;
        mItems.clear();
        mAdapter.notifyDataSetChanged();
        loadPage(false);
    }

    private void loadNextPage() {
        if (mEndReached || mIsLoadingPage) {
            return;
        }
        if (mBrowseMode) {
            browsePage(true);
            return;
        }
        if (mCurrentQuery == null || mCurrentQuery.isEmpty()) {
            return;
        }
        loadPage(true);
    }

    private void browseFolder(@Nullable final String parentId, @Nullable final String title) {
        mCurrentBrowseParent = parentId;
        mNextPageToken = null;
        mIsLoadingPage = false;
        mEndReached = false;
        mItems.clear();
        mAdapter.notifyDataSetChanged();
        if (title != null) {
            mBrowseStack.add(new BrowseLocation(parentId, title));
        }
        updateModeControls();
        browsePage(false);
    }

    private void browseUp() {
        if (mBrowseStack.isEmpty()) {
            return;
        }
        mBrowseStack.remove(mBrowseStack.size() - 1);
        final BrowseLocation parent = mBrowseStack.isEmpty()
                ? null : mBrowseStack.get(mBrowseStack.size() - 1);
        mCurrentBrowseParent = parent == null ? null : parent.parentId;
        mNextPageToken = null;
        mIsLoadingPage = false;
        mEndReached = false;
        mItems.clear();
        mAdapter.notifyDataSetChanged();
        updateModeControls();
        browsePage(false);
    }

    private void browsePage(final boolean append) {
        mIsLoadingPage = true;
        mStatusView.setText("Loading...");
        final String pageToken = append ? mNextPageToken : null;
        mClient.browse(mSelectedProvider, mCurrentBrowseParent, mMaxBytes, pageToken,
                new MediaProviderClient.BrowseCallback() {
            @Override
            public void onResults(final List<MediaProviderItem> items, final String nextPageToken) {
                mIsLoadingPage = false;
                sortBrowseItems(items);
                if (!append) {
                    mItems.clear();
                }
                final int insertStart = mItems.size();
                mItems.addAll(items);
                if (append) {
                    mAdapter.notifyItemRangeInserted(insertStart, items.size());
                } else {
                    mAdapter.notifyDataSetChanged();
                }
                mNextPageToken = nextPageToken;
                mEndReached = nextPageToken == null || nextPageToken.isEmpty() || items.isEmpty();
                if (mItems.isEmpty()) {
                    mStatusView.setText("Empty folder");
                } else {
                    mStatusView.setText(mItems.size() + " item(s)");
                }
            }

            @Override
            public void onError(final String message) {
                mIsLoadingPage = false;
                mStatusView.setText(message);
                Toast.makeText(mLatinIME, message, Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void loadPage(final boolean append) {
        mIsLoadingPage = true;
        if (!append) {
            mStatusView.setText("Searching...");
        } else if (mItems.isEmpty()) {
            mStatusView.setText("Loading more...");
        }
        final String pageToken = append ? mNextPageToken : null;
        mClient.search(mSelectedProvider, mCurrentQuery, mMaxBytes, pageToken,
                new MediaProviderClient.SearchCallback() {
            @Override
            public void onResults(final List<MediaProviderItem> items, final String nextPageToken) {
                mIsLoadingPage = false;
                if (!append) {
                    mItems.clear();
                }
                final List<MediaProviderItem> newItems = append
                        ? withoutExistingItems(items) : items;
                final int insertStart = mItems.size();
                mItems.addAll(newItems);
                if (append) {
                    mAdapter.notifyItemRangeInserted(insertStart, newItems.size());
                } else {
                    mAdapter.notifyDataSetChanged();
                }
                mNextPageToken = nextPageToken;
                mEndReached = nextPageToken == null || nextPageToken.isEmpty()
                        || nextPageToken.equals(pageToken) || newItems.isEmpty();
                if (mItems.isEmpty()) {
                    mStatusView.setText("No results");
                } else if (mEndReached) {
                    mStatusView.setText(mItems.size() + " results");
                } else {
                    mStatusView.setText(mItems.size() + " results, more available");
                }
                maybePrefetchSearchPage();
            }

            @Override
            public void onError(final String message) {
                mIsLoadingPage = false;
                mStatusView.setText(message);
                Toast.makeText(mLatinIME, message, Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void maybePrefetchSearchPage() {
        if (mBrowseMode || mIsLoadingPage || mEndReached) {
            return;
        }
        if (mItems.size() < SEARCH_PREFETCH_TARGET_ITEMS) {
            loadPage(true);
        }
    }

    private List<MediaProviderItem> withoutExistingItems(final List<MediaProviderItem> items) {
        final ArrayList<MediaProviderItem> filtered = new ArrayList<>();
        for (final MediaProviderItem item : items) {
            if (!hasItem(item.id)) {
                filtered.add(item);
            }
        }
        return filtered;
    }

    private boolean hasItem(final String id) {
        for (final MediaProviderItem item : mItems) {
            if (item.id.equals(id)) {
                return true;
            }
        }
        return false;
    }

    private void selectItem(final MediaProviderItem item) {
        if (item.isFolder) {
            browseFolder(item.id, item.title == null ? "Folder" : item.title);
            return;
        }
        mStatusView.setText("Loading media...");
        mClient.getContent(mSelectedProvider, item.id, mMaxBytes, new MediaProviderClient.ContentCallback() {
            @Override
            public void onContent(final MediaProviderItem contentItem) {
                dismiss();
                mLatinIME.insertExternalMedia(contentItem);
            }

            @Override
            public void onError(final String message) {
                mStatusView.setText(message);
                Toast.makeText(mLatinIME, message, Toast.LENGTH_SHORT).show();
            }
        });
    }

    public void dismiss() {
        if (mPopupWindow != null && mPopupWindow.isShowing()) {
            mPopupWindow.dismiss();
        }
        cleanupAfterDismiss();
    }

    private void cleanupAfterDismiss() {
        if (mDismissed) {
            return;
        }
        mDismissed = true;
        mHandler.removeCallbacks(mCursorBlinkRunnable);
        mClient.close();
        mPreviewExecutor.shutdownNow();
        mLatinIME.clearActiveMediaPickerPopup(this);
    }

    private int dp(final float density, final int value) {
        return (int) (value * density + 0.5f);
    }

    private final class MediaAdapter extends RecyclerView.Adapter<MediaViewHolder> {
        @Override
        public MediaViewHolder onCreateViewHolder(final ViewGroup parent, final int viewType) {
            final ImageView imageView = new ImageView(parent.getContext());
            imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
            imageView.setAdjustViewBounds(false);
            imageView.setBackgroundColor(mColors.get(ColorType.KEY_BACKGROUND));
            final int size = parent.getResources().getDisplayMetrics().widthPixels / GRID_SPAN_COUNT;
            final LinearLayout container = new LinearLayout(parent.getContext());
            container.setOrientation(LinearLayout.VERTICAL);
            container.setBackgroundColor(mColors.get(ColorType.KEY_BACKGROUND));
            container.setLayoutParams(new RecyclerView.LayoutParams(size, size));
            imageView.setLayoutParams(new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
            final TextView titleView = new TextView(parent.getContext());
            titleView.setSingleLine(true);
            titleView.setGravity(Gravity.CENTER);
            titleView.setTextSize(11);
            titleView.setTextColor(mColors.get(ColorType.KEY_TEXT));
            titleView.setLayoutParams(new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, dp(
                            parent.getResources().getDisplayMetrics().density, 24)));
            container.addView(imageView);
            container.addView(titleView);
            return new MediaViewHolder(container, imageView, titleView);
        }

        @Override
        public void onBindViewHolder(final MediaViewHolder holder, final int position) {
            final MediaProviderItem item = mItems.get(position);
            cancelPreview(holder);
            holder.imageView.setTag(item.id);
            holder.previewKey = null;
            holder.imageView.setImageDrawable(null);
            holder.imageView.setImageURI(null);
            if (item.title == null || item.title.isEmpty()) {
                holder.titleView.setVisibility(View.GONE);
                holder.titleView.setText("");
            } else {
                holder.titleView.setVisibility(View.VISIBLE);
                holder.titleView.setText(item.title);
            }
            if (item.isFolder) {
                holder.imageView.setImageDrawable(folderDrawable());
            } else if (item.previewUri != null) {
                loadPreview(holder, item);
            }
            holder.itemView.setOnClickListener(view -> selectItem(item));
        }

        @Override
        public int getItemCount() {
            return mItems.size();
        }

        private void loadPreview(final MediaViewHolder holder, final MediaProviderItem item) {
            final String previewKey = item.previewUri.toString();
            final String cacheKey = previewCacheKey(item);
            holder.previewKey = previewKey;
            final Drawable cachedDrawable = cachedPreviewDrawable(cacheKey);
            if (cachedDrawable != null) {
                holder.imageView.setImageDrawable(cachedDrawable);
                startIfAnimated(cachedDrawable);
                return;
            }
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
                holder.imageView.setImageURI(item.previewUri);
                return;
            }
            holder.previewFuture = mPreviewExecutor.submit(() -> {
                try {
                    if (Thread.currentThread().isInterrupted()
                            || !previewKey.equals(holder.previewKey)) {
                        return;
                    }
                    final ImageDecoder.Source source =
                            ImageDecoder.createSource(mLatinIME.getContentResolver(), item.previewUri);
                    final Drawable drawable = ImageDecoder.decodeDrawable(source);
                    mHandler.post(() -> {
                        if (!previewKey.equals(holder.previewKey)) {
                            return;
                        }
                        final Drawable.ConstantState state = drawable.getConstantState();
                        if (state != null) {
                            mPreviewCache.put(cacheKey, state);
                        }
                        holder.imageView.setImageDrawable(drawable);
                        startIfAnimated(drawable);
                    });
                } catch (Throwable ignored) {
                    mHandler.post(() -> {
                        if (previewKey.equals(holder.previewKey)) {
                            holder.imageView.setImageDrawable(null);
                        }
                    });
                }
            });
        }

        @Override
        public void onViewRecycled(final MediaViewHolder holder) {
            cancelPreview(holder);
            holder.previewKey = null;
            holder.imageView.setImageDrawable(null);
            holder.imageView.setImageURI(null);
            super.onViewRecycled(holder);
        }

        private void cancelPreview(final MediaViewHolder holder) {
            if (holder.previewFuture != null) {
                holder.previewFuture.cancel(true);
                holder.previewFuture = null;
                mPreviewExecutor.purge();
            }
        }

        private Drawable cachedPreviewDrawable(final String cacheKey) {
            final Drawable.ConstantState state = mPreviewCache.get(cacheKey);
            return state == null ? null : state.newDrawable();
        }

        private void startIfAnimated(final Drawable drawable) {
            if (drawable instanceof AnimatedImageDrawable) {
                ((AnimatedImageDrawable) drawable).start();
            }
        }
    }

    private static final class MediaViewHolder extends RecyclerView.ViewHolder {
        final ImageView imageView;
        final TextView titleView;
        String previewKey;
        Future<?> previewFuture;

        MediaViewHolder(final View itemView, final ImageView imageView, final TextView titleView) {
            super(itemView);
            this.imageView = imageView;
            this.titleView = titleView;
        }
    }

    private Drawable folderDrawable() {
        final Drawable drawable = ContextCompat.getDrawable(mLatinIME, R.drawable.ic_media_folder);
        if (drawable != null) {
            drawable.mutate().setTint(mColors.get(ColorType.KEY_TEXT));
        }
        return drawable;
    }

    private void sortBrowseItems(final List<MediaProviderItem> items) {
        Collections.sort(items, new Comparator<MediaProviderItem>() {
            @Override
            public int compare(final MediaProviderItem left, final MediaProviderItem right) {
                if (left.isFolder != right.isFolder) {
                    return left.isFolder ? -1 : 1;
                }
                return normalizedTitle(left).compareTo(normalizedTitle(right));
            }
        });
    }

    private String normalizedTitle(final MediaProviderItem item) {
        final String title = item.title == null || item.title.isEmpty() ? item.id : item.title;
        return title == null ? "" : title.toLowerCase(Locale.ROOT);
    }

    private String previewCacheKey(final MediaProviderItem item) {
        final String scope = mBrowseMode ? nullToEmpty(mCurrentBrowseParent)
                : nullToEmpty(mCurrentQuery);
        return mSelectedProvider.key + "|" + mSelectedMode + "|" + scope + "|"
                + item.previewUri;
    }

    private String nullToEmpty(@Nullable final String value) {
        return value == null ? "" : value;
    }

    private String currentBrowseTitle() {
        if (mBrowseStack.isEmpty()) {
            return "Browse media";
        }
        return mBrowseStack.get(mBrowseStack.size() - 1).title;
    }

    private String modeLabel(final String mode) {
        return MODE_BROWSE.equals(mode) ? "Browse" : "Search";
    }

    private static final class BrowseLocation {
        final String parentId;
        final String title;

        BrowseLocation(final String parentId, final String title) {
            this.parentId = parentId;
            this.title = title;
        }
    }

    private final class ProviderChoice {
        final MediaProviderInfo provider;
        final String mode;

        ProviderChoice(final MediaProviderInfo provider, final String mode) {
            this.provider = provider;
            this.mode = mode;
        }

        String label() {
            return provider.label + " " + modeLabel(mode);
        }
    }

}
