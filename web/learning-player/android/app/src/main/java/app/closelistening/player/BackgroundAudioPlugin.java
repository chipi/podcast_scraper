package app.closelistening.player;

import android.media.MediaRouter2;
import android.os.Build;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

/**
 * Capacitor bridge for Android playback: the media session + notification ({@link NowPlaying},
 * {@link PlaybackService}) and the system output switcher (#1310, operator 2026-10-05).
 *
 * JS (services/native.ts) mirrors the player store into it — {@code start} on play, {@code pause}
 * on pause, {@code stop} when the episode ends or is replaced, {@code update} with metadata and
 * position — and listens for {@code action} events: what the lock screen, the notification and
 * headphone buttons asked for, named as the web MediaSession names them.
 */
@CapacitorPlugin(name = "BackgroundAudio")
public class BackgroundAudioPlugin extends Plugin {

    @Override
    public void load() {
        NowPlaying.get(getContext()).listener = (action, seekTime) -> {
            JSObject data = new JSObject();
            data.put("action", action);
            if (seekTime != null) data.put("seekTime", seekTime);
            notifyListeners("action", data);
        };
    }

    @PluginMethod
    public void start(PluginCall call) {
        NowPlaying.get(getContext()).update(null, null, null, null, true, null, null, null);
        PlaybackService.start(getContext());
        call.resolve();
    }

    @PluginMethod
    public void pause(PluginCall call) {
        NowPlaying.get(getContext()).update(null, null, null, null, false, null, null, null);
        PlaybackService.pause(getContext());
        call.resolve();
    }

    @PluginMethod
    public void stop(PluginCall call) {
        PlaybackService.stop(getContext());
        call.resolve();
    }

    @PluginMethod
    public void update(PluginCall call) {
        NowPlaying.get(getContext()).update(
            call.getString("title"),
            call.getString("artist"),
            call.getString("album"),
            call.getString("artworkUrl"),
            call.getBoolean("playing"),
            call.getDouble("position"),
            call.getDouble("duration"),
            call.getDouble("rate"));
        call.resolve();
    }

    /** The system output switcher — Android's counterpart to the AirPlay sheet — is API 34+. */
    @PluginMethod
    public void canShowOutputSwitcher(PluginCall call) {
        JSObject ret = new JSObject();
        ret.put("available", Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE);
        call.resolve(ret);
    }

    @PluginMethod
    public void showOutputSwitcher(PluginCall call) {
        boolean shown = false;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            shown = MediaRouter2.getInstance(getContext()).showSystemOutputSwitcher();
        }
        JSObject ret = new JSObject();
        ret.put("shown", shown);
        call.resolve(ret);
    }
}
