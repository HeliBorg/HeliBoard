/*
 * SwipeGestureEngine - gesture path matching for HeliBoard.
 *
 * Algorithm: arc-length resampling + L2 distance scoring.
 * Each word in the dictionary is pre-mapped to a path of N_PTS evenly-spaced
 * (x, y) points (normalized to keyboard dimensions).  On gesture end the input
 * stroke is resampled the same way and candidates are ranked by L2 distance
 * with a small log-frequency bonus.
 *
 * Usage (from Suggest.kt):
 *
 *   // Build once per dictionary/layout change (call on background thread):
 *   val index = SwipeGestureEngine.buildIndex(
 *       mDictionaryFacilitator.getAllMainDictionaryWordsWithFrequency(), keyboard)
 *
 *   // On each gesture end:
 *   val results = SwipeGestureEngine.rankByIndex(index, pointers, keyboard, maxResults)
 */
package helium314.keyboard.latin.gesture;

import helium314.keyboard.keyboard.Key;
import helium314.keyboard.keyboard.Keyboard;
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo;
import helium314.keyboard.latin.common.InputPointers;
import helium314.keyboard.latin.dictionary.Dictionary;
import helium314.keyboard.latin.utils.SuggestionResults;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class SwipeGestureEngine {

    /** Number of evenly-spaced points each gesture path is resampled to. */
    private static final int N_PTS = 16;

    /** Weight applied to log(frequency) when scoring candidates. */
    private static final float FREQ_WEIGHT = 0.05f;

    // ── Precomputed index ─────────────────────────────────────────────────────

    /** Precomputed path data for a single dictionary word. */
    public static class IndexEntry {
        public final String word;
        public final float[] path; // length N_PTS*2
        public final int frequency; // raw probability from binary dictionary
        IndexEntry(String word, float[] path, int frequency) { this.word = word; this.path = path; this.frequency = frequency; }
    }

    /**
     * Index of all dictionary words grouped by first letter, with paths precomputed
     * for a specific keyboard layout.  Build once; reuse across gestures.
     */
    public static class GestureIndex {
        public final Map<Character, List<IndexEntry>> byFirst;
        GestureIndex(Map<Character, List<IndexEntry>> byFirst) {
            this.byFirst = byFirst;
        }
    }

    /**
     * Precompute gesture paths for every word in {@code wordsWithFreq} using the given keyboard
     * layout. Call this on a background thread after a dictionary or layout change.
     *
     * @param wordsWithFreq map of word → raw probability from the binary dictionary
     * @param keyboard      current keyboard (determines key positions)
     */
    public static GestureIndex buildIndex(Map<String, Integer> wordsWithFreq, Keyboard keyboard) {
        float[][] charToPos = buildCharToPos(keyboard);
        Map<Character, List<IndexEntry>> byFirst = new HashMap<>();
        for (Map.Entry<String, Integer> entry : wordsWithFreq.entrySet()) {
            String raw = entry.getKey();
            int freq = entry.getValue() != null ? entry.getValue() : 0;
            String word = raw.toLowerCase(Locale.ROOT);
            if (word.isEmpty()) continue;
            char first = word.charAt(0);
            if (first < 'a' || first > 'z') continue;
            float[] path = wordPath(word, charToPos);
            byFirst.computeIfAbsent(first, k -> new ArrayList<>())
                    .add(new IndexEntry(raw, path, freq));
        }
        return new GestureIndex(byFirst);
    }

    /**
     * Fingerprint of the key positions for a given keyboard layout.
     * Stable across shift-state and action-button changes; changes only when key centres move
     * (i.e. when the user switches language or physical layout).  Used by Suggest.kt to decide
     * whether to rebuild the index.
     */
    public static int layoutFingerprint(Keyboard keyboard) {
        return Arrays.deepHashCode(buildCharToPos(keyboard));
    }

    // ── Public matching API ───────────────────────────────────────────────────

    /** Return true if code is an ASCII letter (upper or lower case). */
    private static boolean isAsciiLetter(int code) {
        return (code >= 'a' && code <= 'z') || (code >= 'A' && code <= 'Z');
    }

    public static char nearestLetter(int x, int y, Keyboard keyboard) {
        // Always use a full distance scan — getNearestKeys uses a proximity grid that is
        // NOT sorted by distance, so the first letter it returns is often wrong.
        float kw = keyboard.mOccupiedWidth, kh = keyboard.mOccupiedHeight;
        float nx = x / kw, ny = y / kh;
        float minDist = Float.MAX_VALUE;
        char best = 0;
        for (Key key : keyboard.getSortedKeys()) {
            int code = key.getCode();
            if (!isAsciiLetter(code)) continue;
            float cx = (key.getX() + key.getWidth()  / 2f) / kw;
            float cy = (key.getY() + key.getHeight() / 2f) / kh;
            float d  = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy);
            if (d < minDist) { minDist = d; best = Character.toLowerCase((char) code); }
        }
        return best;
    }

    /**
     * Match the gesture stroke against the precomputed index.
     *
     * <ol>
     *   <li>Detect first and last letters from gesture endpoints.</li>
     *   <li>Look up all words in the index that start with the first letter.</li>
     *   <li>Filter by last letter (relaxed if that leaves nothing).</li>
     *   <li>Resample the input stroke and rank candidates by L2 distance to their precomputed path.</li>
     * </ol>
     *
     * @param index      precomputed word index (from {@link #buildIndex})
     * @param pointers   raw pixel touch coordinates from BatchInputArbiter
     * @param keyboard   current keyboard
     * @param maxResults maximum results to return
     */
    public static SuggestionResults rankByIndex(
            GestureIndex index,
            InputPointers pointers,
            Keyboard keyboard,
            int maxResults
    ) {
        int n = pointers.getPointerSize();
        SuggestionResults empty = new SuggestionResults(1, false, false);
        if (n < 2 || index == null) return empty;

        int[] xs = pointers.getXCoordinates();
        int[] ys = pointers.getYCoordinates();

        float kw = keyboard.mOccupiedWidth, kh = keyboard.mOccupiedHeight;

        char firstLetter = nearestLetter(xs[0],     ys[0],     keyboard);
        char lastLetter  = nearestLetter(xs[n - 1], ys[n - 1], keyboard);

        List<IndexEntry> candidates = index.byFirst.get(firstLetter);
        if (candidates == null || candidates.isEmpty()) return empty;
        List<float[]> rawPath = new ArrayList<>(n);
        for (int i = 0; i < n; i++) rawPath.add(new float[]{xs[i] / kw, ys[i] / kh});
        float[] inputVec = resample(rawPath, N_PTS);

        // Filter by last letter first; relax if empty
        List<IndexEntry> filtered = new ArrayList<>();
        for (IndexEntry e : candidates) {
            String w = e.word.toLowerCase(Locale.ROOT);
            if (lastLetter != 0 && !w.isEmpty() && w.charAt(w.length() - 1) == lastLetter)
                filtered.add(e);
        }
        if (filtered.isEmpty()) filtered = candidates;

        // Score: negative L2 distance + log-frequency bonus (frequency [0..255], log scales nicely)
        int m = filtered.size();
        float[] scores = new float[m];
        for (int i = 0; i < m; i++) {
            IndexEntry e = filtered.get(i);
            float freqBonus = (e.frequency > 0) ? (float)(Math.log(e.frequency + 1) * FREQ_WEIGHT) : 0f;
            scores[i] = -l2(inputVec, e.path) + freqBonus;
        }

        Integer[] idx = new Integer[m];
        for (int i = 0; i < m; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(scores[b], scores[a]));

        int take = Math.min(maxResults, m);
        SuggestionResults result = new SuggestionResults(take, false, false);
        int baseScore = 1_000_000;
        for (int rank = 0; rank < take; rank++) {
            IndexEntry e = filtered.get(idx[rank]);
            result.add(new SuggestedWordInfo(
                    e.word, "",
                    baseScore - rank * 1000,
                    SuggestedWordInfo.KIND_CORRECTION,
                    Dictionary.DICTIONARY_USER_TYPED,
                    SuggestedWordInfo.NOT_AN_INDEX,
                    SuggestedWordInfo.NOT_A_CONFIDENCE
            ));
        }
        return result;
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    /** Build letter → normalized (x, y) center map from the current keyboard layout. */
    static float[][] buildCharToPos(Keyboard keyboard) {
        float[][] map = new float[26][2];
        float kw = keyboard.mOccupiedWidth, kh = keyboard.mOccupiedHeight;
        for (Key key : keyboard.getSortedKeys()) {
            int code = key.getCode();
            if (!isAsciiLetter(code)) continue;
            int idx = Character.toLowerCase((char) code) - 'a';
            map[idx][0] = (key.getX() + key.getWidth()  / 2f) / kw;
            map[idx][1] = (key.getY() + key.getHeight() / 2f) / kh;
        }
        return map;
    }

    /** Convert a word to its ideal gesture path: letter centers, deduplicated, resampled. */
    static float[] wordPath(String word, float[][] charToPos) {
        List<float[]> pts = new ArrayList<>();
        for (char c : word.toCharArray()) {
            int idx = c - 'a';
            if (idx < 0 || idx >= 26) continue;
            float[] p = charToPos[idx];
            if (pts.isEmpty()
                    || pts.get(pts.size() - 1)[0] != p[0]
                    || pts.get(pts.size() - 1)[1] != p[1]) {
                pts.add(new float[]{p[0], p[1]});
            }
        }
        return resample(pts, N_PTS);
    }

    /**
     * Arc-length resample: convert an arbitrary list of (x,y) points to exactly
     * {@code n} evenly-spaced points by arc length, returned as float[n*2].
     * Makes paths invariant to finger speed.
     */
    static float[] resample(List<float[]> pts, int n) {
        if (pts.isEmpty()) return new float[n * 2];
        if (pts.size() == 1) {
            float[] r = new float[n * 2];
            float x = pts.get(0)[0], y = pts.get(0)[1];
            for (int i = 0; i < n; i++) { r[2*i] = x; r[2*i+1] = y; }
            return r;
        }
        float[] cum = new float[pts.size()];
        for (int i = 1; i < pts.size(); i++) {
            float dx = pts.get(i)[0] - pts.get(i-1)[0];
            float dy = pts.get(i)[1] - pts.get(i-1)[1];
            cum[i] = cum[i-1] + (float) Math.sqrt(dx*dx + dy*dy);
        }
        float total = cum[pts.size()-1];
        if (total < 1e-9f) {
            float[] r = new float[n * 2];
            float x = pts.get(0)[0], y = pts.get(0)[1];
            for (int i = 0; i < n; i++) { r[2*i] = x; r[2*i+1] = y; }
            return r;
        }
        float[] result = new float[n * 2];
        int seg = 0;
        for (int i = 0; i < n; i++) {
            float t = total * i / (n - 1);
            while (seg < pts.size() - 2 && cum[seg + 1] < t) seg++;
            float segLen = cum[seg+1] - cum[seg];
            float alpha  = (segLen > 1e-9f) ? (t - cum[seg]) / segLen : 0f;
            result[2*i]   = pts.get(seg)[0] + alpha * (pts.get(seg+1)[0] - pts.get(seg)[0]);
            result[2*i+1] = pts.get(seg)[1] + alpha * (pts.get(seg+1)[1] - pts.get(seg)[1]);
        }
        return result;
    }

    private static float l2(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) { float d = a[i] - b[i]; s += d*d; }
        return (float) Math.sqrt(s);
    }
}
