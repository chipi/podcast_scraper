package app.closelistening.player;

import android.os.Process;
import android.os.SystemClock;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

/**
 * How long THIS app process has been alive (#2277) — the Android twin of iOS AppProcess in
 * MainViewController.swift. A page that boots inside a process much older than itself is a
 * WebView reload, not a cold launch.
 */
@CapacitorPlugin(name = "AppProcess")
public class AppProcessPlugin extends Plugin {
    @PluginMethod
    public void uptime(PluginCall call) {
        JSObject result = new JSObject();
        result.put("ms", SystemClock.elapsedRealtime() - Process.getStartElapsedRealtime());
        call.resolve(result);
    }
}
