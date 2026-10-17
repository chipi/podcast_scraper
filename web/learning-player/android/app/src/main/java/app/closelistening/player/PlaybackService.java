package app.closelistening.player;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;
import androidx.annotation.Nullable;

/**
 * The media notification and the foreground keep-alive for background playback (#1310).
 *
 * Android suspends a backgrounded app's media unless it runs a foreground service of type
 * mediaPlayback. This used to be ONLY that: a bare "Playing in the background" notification, on the
 * assumption that the WebView's MediaSession would supply the real transport. Android System WebView
 * does not (operator 2026-10-05: no lock-screen controls, no output switcher), so the notification is
 * now the media notification itself — artwork, title, show, back 15 / play-pause / forward 30 — bound
 * to {@link NowPlaying}'s session, which is what the lock screen and the system output switcher read.
 *
 * Paused, it stays (detached from the foreground) so the listener can resume from the lock screen;
 * {@link #stop} removes it when the episode ends or is replaced.
 */
public class PlaybackService extends Service {

    private static final String TAG = "PlaybackService";
    private static final String CHANNEL_ID = "lp_playback";
    private static final int NOTIFICATION_ID = 1310;

    static final String MODE = "mode";
    static final String MODE_PLAYING = "playing";
    static final String MODE_PAUSED = "paused";
    /** Notification buttons arrive as these, and are relayed as the web MediaSession's actions. */
    static final String BUTTON = "button";

    /**
     * The service while it runs. A play or pause after the first is handed to it directly. Each
     * used to be another {@code startForegroundService()}, which Android 12+ refuses while the app
     * is in the background: playback paused (the service left the foreground), the app went to the
     * background, something resumed it, and the refused start threw out of the plugin call and
     * killed the app (PLAYER-1I, Android 16, 2026-10-17; reproduced on the emulator).
     */
    private static volatile PlaybackService running;

    @Override
    public void onCreate() {
        super.onCreate();
        running = this;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "Playback",
                NotificationManager.IMPORTANCE_LOW // silent: it carries controls, it is not an alert
            );
            channel.setDescription("Playback controls while listening");
            channel.setLockscreenVisibility(Notification.VISIBILITY_PUBLIC);
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) manager.createNotificationChannel(channel);
        }
        NowPlaying.get(this).onChanged = this::redraw;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        NowPlaying np = NowPlaying.get(this);
        String button = intent == null ? null : intent.getStringExtra(BUTTON);
        if (button != null) {
            // The notification's Play is a foreground-service PendingIntent: a tap on a notification
            // is one of the cases Android lets start a foreground service from the background, so
            // playback is in the foreground before the player even hears the button.
            if ("play".equals(button)) promote();
            np.emit(button, null);
            return START_NOT_STICKY;
        }
        String mode = intent == null ? MODE_PLAYING : intent.getStringExtra(MODE);
        promote();
        if (MODE_PAUSED.equals(mode)) demote();
        // START_NOT_STICKY: if the OS kills us, don't resurrect a service with no live playback.
        return START_NOT_STICKY;
    }

    /**
     * The media notification, in the foreground. False when Android refuses (the app is in the
     * background and no exemption applies): the notification is still shown, as playing, and the
     * audio goes on while the process lives.
     */
    private boolean promote() {
        NowPlaying np = NowPlaying.get(this);
        np.setActive(true);
        Notification notification = buildNotification(np);
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                // Android 14+ requires the type at startForeground time, matching the manifest.
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK);
            } else {
                startForeground(NOTIFICATION_ID, notification);
            }
            return true;
        } catch (IllegalStateException e) { // ForegroundServiceStartNotAllowedException, API 31+
            Log.w(TAG, "foreground refused; playing without it: " + e.getMessage());
            redraw();
            return false;
        }
    }

    /** Paused: the controls stay on screen, but the OS may treat the process as idle. */
    private void demote() {
        stopForeground(STOP_FOREGROUND_DETACH);
        redraw();
    }

    @Override
    public void onDestroy() {
        if (running == this) running = null;
        NowPlaying np = NowPlaying.get(this);
        np.onChanged = null;
        np.setActive(false);
        super.onDestroy();
    }

    private void redraw() {
        NotificationManager manager = getSystemService(NotificationManager.class);
        if (manager != null) manager.notify(NOTIFICATION_ID, buildNotification(NowPlaying.get(this)));
    }

    private PendingIntent button(String action, int requestCode) {
        Intent i = new Intent(this, PlaybackService.class).putExtra(BUTTON, action);
        int flags = PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT;
        // Play starts playback, so it starts this service as a FOREGROUND service: allowed from a
        // notification tap even while the app is in the background, and then onStartCommand must
        // (and does) call startForeground. The other buttons do not need the foreground.
        if ("play".equals(action) && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            return PendingIntent.getForegroundService(this, requestCode, i, flags);
        }
        return PendingIntent.getService(this, requestCode, i, flags);
    }

    @SuppressWarnings("deprecation") // addAction(int, …): the Icon overload needs API 23 for nothing gained
    private Notification buildNotification(NowPlaying np) {
        Intent launch = getPackageManager().getLaunchIntentForPackage(getPackageName());
        PendingIntent contentIntent = PendingIntent.getActivity(
            this, 0, launch, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        Notification.Builder b = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
            ? new Notification.Builder(this, CHANNEL_ID)
            : new Notification.Builder(this);
        b.setSmallIcon(android.R.drawable.ic_media_play)
            .setContentTitle(np.title.isEmpty() ? getString(R.string.app_name) : np.title)
            .setContentText(np.artist)
            .setContentIntent(contentIntent)
            .setVisibility(Notification.VISIBILITY_PUBLIC)
            .setOngoing(np.playing)
            .setShowWhen(false)
            .addAction(android.R.drawable.ic_media_rew, "Back 15 seconds", button("seekbackward", 1))
            .addAction(
                np.playing ? android.R.drawable.ic_media_pause : android.R.drawable.ic_media_play,
                np.playing ? "Pause" : "Play",
                button(np.playing ? "pause" : "play", 2))
            .addAction(android.R.drawable.ic_media_ff, "Forward 30 seconds", button("seekforward", 3))
            .setStyle(new Notification.MediaStyle()
                .setMediaSession(np.session.getSessionToken())
                .setShowActionsInCompactView(0, 1, 2));
        if (np.artwork != null) b.setLargeIcon(np.artwork);
        return b.build();
    }

    /** Playing: the media notification, in the foreground. False when Android refused that. */
    public static boolean start(Context context) {
        PlaybackService s = running;
        return s != null ? s.promote() : send(context, MODE_PLAYING);
    }

    /** Paused: the notification stays, with Play, so the lock screen can resume. */
    public static boolean pause(Context context) {
        PlaybackService s = running;
        if (s == null) return send(context, MODE_PAUSED);
        s.demote(); // leaving the foreground needs no permission
        return true;
    }

    /** Start the service (it then promotes itself); false when Android refused the start. */
    private static boolean send(Context context, String mode) {
        Intent intent = new Intent(context, PlaybackService.class).putExtra(MODE, mode);
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent);
            } else {
                context.startService(intent);
            }
            return true;
        } catch (IllegalStateException e) { // ForegroundServiceStartNotAllowedException, API 31+
            Log.w(TAG, "playback service start refused: " + e.getMessage());
            return false;
        }
    }

    /** Episode ended or replaced: notification, session and service all go. */
    public static void stop(Context context) {
        context.stopService(new Intent(context, PlaybackService.class));
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
