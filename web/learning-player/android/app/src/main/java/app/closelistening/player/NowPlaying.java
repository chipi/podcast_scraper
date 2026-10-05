package app.closelistening.player;

import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaMetadata;
import android.media.session.MediaSession;
import android.media.session.PlaybackState;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import androidx.annotation.Nullable;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * The app's Android media session — what the lock screen, the media notification, headphone and
 * Bluetooth buttons, and the system output switcher talk to (operator 2026-10-05).
 *
 * The player plays in the WebView's &lt;audio&gt;, and on iOS WKWebView turns the page's
 * {@code navigator.mediaSession} into Now Playing by itself. Android System WebView does not: the
 * keep-alive service showed a bare "Playing in the background" notification, the lock screen had no
 * controls, and nothing gave the system output switcher a session to route. So the session lives
 * here, natively, fed by the player store over {@link BackgroundAudioPlugin}, and every control
 * press is relayed back to the store as the same action names the web MediaSession handlers use —
 * one set of handlers, two platforms.
 *
 * Framework classes only ({@code android.media.session}, {@code Notification.MediaStyle}): no new
 * dependency.
 */
final class NowPlaying {

    /** An action from the system controls, named as the web MediaSession names them. */
    interface Listener {
        void onAction(String action, @Nullable Double seekTime);
    }

    private static NowPlaying instance;

    static synchronized NowPlaying get(Context context) {
        if (instance == null) instance = new NowPlaying(context.getApplicationContext());
        return instance;
    }

    final MediaSession session;
    private final Context context;
    private final Handler main = new Handler(Looper.getMainLooper());

    String title = "";
    String artist = "";
    String album = "";
    @Nullable Bitmap artwork;
    private @Nullable String artworkUrl;
    long durationMs = 0;
    long positionMs = 0;
    float rate = 1f;
    boolean playing = false;

    @Nullable Listener listener;
    /** The service redraws its notification on any change (set while the service is alive). */
    @Nullable Runnable onChanged;

    private NowPlaying(Context context) {
        this.context = context;
        session = new MediaSession(context, "CloseListening");
        Intent launch = context.getPackageManager().getLaunchIntentForPackage(context.getPackageName());
        if (launch != null) {
            session.setSessionActivity(PendingIntent.getActivity(
                context, 0, launch, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT));
        }
        session.setCallback(new MediaSession.Callback() {
            @Override public void onPlay() { emit("play", null); }
            @Override public void onPause() { emit("pause", null); }
            @Override public void onStop() { emit("pause", null); }
            @Override public void onRewind() { emit("seekbackward", null); }
            @Override public void onFastForward() { emit("seekforward", null); }
            @Override public void onSkipToNext() { emit("nexttrack", null); }
            @Override public void onSkipToPrevious() { emit("previoustrack", null); }
            @Override public void onSeekTo(long pos) { emit("seekto", pos / 1000.0); }
        });
    }

    void emit(String action, @Nullable Double seekTime) {
        Listener l = listener;
        if (l != null) l.onAction(action, seekTime);
    }

    /** Apply whatever the store sent; absent fields keep their value. */
    void update(@Nullable String title, @Nullable String artist, @Nullable String album,
                @Nullable String artworkUrl, @Nullable Boolean playing, @Nullable Double position,
                @Nullable Double duration, @Nullable Double rate) {
        boolean metadata = false;
        if (title != null) { this.title = title; metadata = true; }
        if (artist != null) { this.artist = artist; metadata = true; }
        if (album != null) { this.album = album; metadata = true; }
        if (duration != null && duration > 0) { this.durationMs = (long) (duration * 1000); metadata = true; }
        if (playing != null) this.playing = playing;
        if (position != null && position >= 0) this.positionMs = (long) (position * 1000);
        if (rate != null && rate > 0) this.rate = rate.floatValue();
        if (artworkUrl != null && !artworkUrl.equals(this.artworkUrl)) {
            this.artworkUrl = artworkUrl;
            this.artwork = null;
            metadata = true;
            loadArtwork(artworkUrl);
        }
        if (metadata) pushMetadata();
        pushState();
        changed();
    }

    void setActive(boolean active) {
        session.setActive(active);
    }

    private void pushMetadata() {
        MediaMetadata.Builder b = new MediaMetadata.Builder()
            .putString(MediaMetadata.METADATA_KEY_TITLE, title)
            .putString(MediaMetadata.METADATA_KEY_ARTIST, artist)
            .putString(MediaMetadata.METADATA_KEY_ALBUM, album)
            .putLong(MediaMetadata.METADATA_KEY_DURATION, durationMs);
        if (artwork != null) b.putBitmap(MediaMetadata.METADATA_KEY_ART, artwork);
        session.setMetadata(b.build());
    }

    private void pushState() {
        long actions = PlaybackState.ACTION_PLAY | PlaybackState.ACTION_PAUSE
            | PlaybackState.ACTION_PLAY_PAUSE | PlaybackState.ACTION_SEEK_TO
            | PlaybackState.ACTION_REWIND | PlaybackState.ACTION_FAST_FORWARD
            | PlaybackState.ACTION_SKIP_TO_NEXT | PlaybackState.ACTION_SKIP_TO_PREVIOUS
            | PlaybackState.ACTION_STOP;
        session.setPlaybackState(new PlaybackState.Builder()
            .setActions(actions)
            .setState(
                playing ? PlaybackState.STATE_PLAYING : PlaybackState.STATE_PAUSED,
                positionMs,
                playing ? rate : 0f,
                SystemClock.elapsedRealtime())
            .build());
    }

    private void changed() {
        Runnable r = onChanged;
        if (r != null) main.post(r);
    }

    /** Off the main thread; the cover is a nicety, so any failure just leaves it out. */
    private void loadArtwork(String url) {
        new Thread(() -> {
            Bitmap bmp = null;
            HttpURLConnection conn = null;
            try {
                conn = (HttpURLConnection) new URL(url).openConnection();
                conn.setConnectTimeout(8_000);
                conn.setReadTimeout(8_000);
                try (InputStream in = conn.getInputStream()) {
                    bmp = BitmapFactory.decodeStream(in);
                }
                if (bmp != null && Math.max(bmp.getWidth(), bmp.getHeight()) > 512) {
                    float s = 512f / Math.max(bmp.getWidth(), bmp.getHeight());
                    bmp = Bitmap.createScaledBitmap(
                        bmp, Math.round(bmp.getWidth() * s), Math.round(bmp.getHeight() * s), true);
                }
            } catch (Exception ignored) {
                // no artwork rather than no controls
            } finally {
                if (conn != null) conn.disconnect();
            }
            final Bitmap result = bmp;
            main.post(() -> {
                if (!url.equals(artworkUrl) || result == null) return;
                artwork = result;
                pushMetadata();
                changed();
            });
        }, "now-playing-art").start();
    }
}
