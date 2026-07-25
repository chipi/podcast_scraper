package app.closelistening.player;

import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

/**
 * Capacitor bridge for the background-audio keep-alive (#1310). JS (services/native.ts) calls
 * BackgroundAudio.start() when playback begins and .stop() when it pauses/ends, so the
 * {@link PlaybackService} foreground service runs only while audio is actually playing.
 */
@CapacitorPlugin(name = "BackgroundAudio")
public class BackgroundAudioPlugin extends Plugin {

    @PluginMethod
    public void start(PluginCall call) {
        PlaybackService.start(getContext());
        call.resolve();
    }

    @PluginMethod
    public void stop(PluginCall call) {
        PlaybackService.stop(getContext());
        call.resolve();
    }
}
