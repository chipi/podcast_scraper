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

    private static final String CHANNEL_ID = "lp_playback";
    private static final int NOTIFICATION_ID = 1310;

    static final String MODE = "mode";
    static final String MODE_PLAYING = "playing";
    static final String MODE_PAUSED = "paused";
    /** Notification buttons arrive as these, and are relayed as the web MediaSession's actions. */
    static final String BUTTON = "button";

    @Override
    public void onCreate() {
        super.onCreate();
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
            np.emit(button, null);
            return START_NOT_STICKY;
        }
        String mode = intent == null ? MODE_PLAYING : intent.getStringExtra(MODE);
        np.setActive(true);
        Notification notification = buildNotification(np);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            // Android 14+ requires the type at startForeground time, matching the manifest.
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
        if (MODE_PAUSED.equals(mode)) {
            // Keep the controls on screen, but let the OS treat the process as idle while paused.
            stopForeground(STOP_FOREGROUND_DETACH);
        }
        // START_NOT_STICKY: if the OS kills us, don't resurrect a service with no live playback.
        return START_NOT_STICKY;
    }

    @Override
    public void onDestroy() {
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
        return PendingIntent.getService(
            this, requestCode, i, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
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

    /** Playing: the media notification, in the foreground. */
    public static void start(Context context) {
        send(context, MODE_PLAYING);
    }

    /** Paused: the notification stays, with Play, so the lock screen can resume. */
    public static void pause(Context context) {
        send(context, MODE_PAUSED);
    }

    private static void send(Context context, String mode) {
        Intent intent = new Intent(context, PlaybackService.class).putExtra(MODE, mode);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(intent);
        } else {
            context.startService(intent);
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
