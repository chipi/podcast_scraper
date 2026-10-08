package app.closelistening.player;

import android.app.ActivityManager;
import android.app.ApplicationExitInfo;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.os.Build;
import android.os.Debug;
import android.os.PowerManager;
import android.os.Process;
import android.os.SystemClock;
import android.webkit.RenderProcessGoneDetail;
import android.webkit.WebView;
import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.WebViewListener;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.TimeZone;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * The app process as the web layer cannot see it — the Android twin of iOS AppProcess in
 * MainViewController.swift.
 *
 * `uptime` (#2277): how long THIS process has been alive; a page booting inside a much older
 * process is a WebView reload, not a cold launch.
 *
 * `memoryInfo` (operator 2026-10-08): the memory and device facts "Copy debug info" in Settings
 * cannot read from the page — free / total RAM, the low-memory flag, this process's footprint,
 * thermal status and the WebView package actually in use. NOT included, because Android offers no
 * API for it: free GPU memory. The WebView renderer is a separate process, so its memory is not in
 * `appPssMb` either.
 *
 * `exitLog` / `clearExitLog` (#2279): why the app or its WebView ended — the system's own
 * {@link ApplicationExitInfo} history (Android 11+) and the WebView renderer dying — so the web
 * layer can forward it to /api/app/app-exits on the next launch.
 */
@CapacitorPlugin(name = "AppProcess")
public class AppProcessPlugin extends Plugin {
    private static final String PREFS = "lp.exitLog";
    private static final String KEY_ENTRIES = "entries";
    /** Newest ApplicationExitInfo timestamp already delivered, so history is reported once. */
    private static final String KEY_LAST_EXIT_TS = "lastExitTs";
    private static final int CAP = 50;

    @Override
    public void load() {
        // Recorded with commit(), not apply(): returning false below lets Android kill the app
        // (Capacitor's behaviour, unchanged here), so an asynchronous write would be lost with it.
        bridge.addWebViewListener(
            new WebViewListener() {
                @Override
                public boolean onRenderProcessGone(WebView view, RenderProcessGoneDetail detail) {
                    append("webview_terminated", detail.didCrash() ? "renderer_crashed" : "renderer_killed", 1, new Date());
                    return false;
                }
            }
        );
    }

    @PluginMethod
    public void uptime(PluginCall call) {
        JSObject result = new JSObject();
        result.put("ms", SystemClock.elapsedRealtime() - Process.getStartElapsedRealtime());
        call.resolve(result);
    }

    @PluginMethod
    public void memoryInfo(PluginCall call) {
        JSObject r = new JSObject();
        ActivityManager am = (ActivityManager) getContext().getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        am.getMemoryInfo(mi);
        r.put("availMb", mi.availMem / 1048576);
        r.put("totalMb", mi.totalMem / 1048576);
        r.put("thresholdMb", mi.threshold / 1048576);
        r.put("lowMemory", mi.lowMemory);
        r.put("lowRamDevice", am.isLowRamDevice());
        r.put("memoryClassMb", am.getMemoryClass());
        Debug.MemoryInfo own = new Debug.MemoryInfo();
        Debug.getMemoryInfo(own);
        r.put("appPssMb", own.getTotalPss() / 1024);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            PowerManager pm = (PowerManager) getContext().getSystemService(Context.POWER_SERVICE);
            r.put("thermalStatus", pm.getCurrentThermalStatus());
        }
        r.put("manufacturer", Build.MANUFACTURER);
        r.put("model", Build.MODEL);
        r.put("osVersion", Build.VERSION.RELEASE + " (SDK " + Build.VERSION.SDK_INT + ")");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            PackageInfo wv = WebView.getCurrentWebViewPackage();
            if (wv != null) r.put("webView", wv.packageName + " " + wv.versionName);
        }
        call.resolve(r);
    }

    /** Pending records, oldest first: the stored ones, then system exit history not yet delivered. */
    @PluginMethod
    public void exitLog(PluginCall call) {
        JSArray entries = new JSArray();
        try {
            JSONArray stored = new JSONArray(prefs().getString(KEY_ENTRIES, "[]"));
            for (int i = 0; i < stored.length(); i++) entries.put(stored.getJSONObject(i));
        } catch (JSONException ignored) {
            // A corrupt log is dropped rather than blocking the history below.
        }
        long newest = prefs().getLong(KEY_LAST_EXIT_TS, 0);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            ActivityManager am = (ActivityManager) getContext().getSystemService(Context.ACTIVITY_SERVICE);
            List<ApplicationExitInfo> history = am.getHistoricalProcessExitReasons(getContext().getPackageName(), 0, 20);
            // The list is newest first; report oldest first, like the stored log.
            for (int i = history.size() - 1; i >= 0; i--) {
                ApplicationExitInfo info = history.get(i);
                if (info.getTimestamp() <= prefs().getLong(KEY_LAST_EXIT_TS, 0)) continue;
                entries.put(entry("android_exit_info", reasonName(info.getReason()), 1, new Date(info.getTimestamp())));
                newest = Math.max(newest, info.getTimestamp());
            }
        }
        JSObject result = new JSObject();
        result.put("entries", entries);
        result.put("historyThrough", newest);
        call.resolve(result);
    }

    /** Forget what was delivered: the stored log, and history up to `historyThrough`. */
    @PluginMethod
    public void clearExitLog(PluginCall call) {
        prefs()
            .edit()
            .putString(KEY_ENTRIES, "[]")
            .putLong(KEY_LAST_EXIT_TS, Math.max(prefs().getLong(KEY_LAST_EXIT_TS, 0), call.getLong("historyThrough", 0L)))
            .commit();
        call.resolve();
    }

    private SharedPreferences prefs() {
        return getContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    private synchronized void append(String source, String reason, int count, Date at) {
        try {
            JSONArray list = new JSONArray(prefs().getString(KEY_ENTRIES, "[]"));
            list.put(entry(source, reason, count, at));
            JSONArray capped = new JSONArray();
            for (int i = Math.max(0, list.length() - CAP); i < list.length(); i++) capped.put(list.get(i));
            prefs().edit().putString(KEY_ENTRIES, capped.toString()).commit();
        } catch (JSONException ignored) {
            // Best-effort: a lost record must never break the WebView callback.
        }
    }

    private static JSONObject entry(String source, String reason, int count, Date at) {
        JSONObject o = new JSONObject();
        try {
            o.put("source", source);
            o.put("reason", reason);
            o.put("count", count);
            SimpleDateFormat iso = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US);
            iso.setTimeZone(TimeZone.getTimeZone("UTC"));
            o.put("at", iso.format(at));
        } catch (JSONException ignored) {}
        return o;
    }

    /** ApplicationExitInfo.REASON_* as snake_case; the server accepts ^[a-z0-9_]{1,48}$. */
    static String reasonName(int reason) {
        switch (reason) {
            case ApplicationExitInfo.REASON_EXIT_SELF: return "exit_self";
            case ApplicationExitInfo.REASON_SIGNALED: return "signaled";
            case ApplicationExitInfo.REASON_LOW_MEMORY: return "low_memory";
            case ApplicationExitInfo.REASON_CRASH: return "crash";
            case ApplicationExitInfo.REASON_CRASH_NATIVE: return "crash_native";
            case ApplicationExitInfo.REASON_ANR: return "anr";
            case ApplicationExitInfo.REASON_INITIALIZATION_FAILURE: return "initialization_failure";
            case ApplicationExitInfo.REASON_PERMISSION_CHANGE: return "permission_change";
            case ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE: return "excessive_resource_usage";
            case ApplicationExitInfo.REASON_USER_REQUESTED: return "user_requested";
            case ApplicationExitInfo.REASON_USER_STOPPED: return "user_stopped";
            case ApplicationExitInfo.REASON_DEPENDENCY_DIED: return "dependency_died";
            case ApplicationExitInfo.REASON_OTHER: return "other";
            case ApplicationExitInfo.REASON_FREEZER: return "freezer";
            case ApplicationExitInfo.REASON_PACKAGE_STATE_CHANGE: return "package_state_change";
            case ApplicationExitInfo.REASON_PACKAGE_UPDATED: return "package_updated";
            default: return "unknown";
        }
    }
}
