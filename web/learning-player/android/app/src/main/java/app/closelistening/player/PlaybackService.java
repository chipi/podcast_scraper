package app.closelistening.player;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;

/**
 * Foreground service that keeps the process alive while the WebView's &lt;audio&gt; plays in the
 * background (#1310). Android suspends a backgrounded app's media unless it runs a foreground
 * service of type mediaPlayback; this service is that keep-alive + the required ongoing
 * notification. The rich transport (play/pause/seek, artwork) is provided separately by the
 * WebView's MediaSession (wired in the Pinia player store, #1308) — this service does not itself
 * play audio. Started/stopped from JS via {@link BackgroundAudioPlugin} on play/pause.
 */
public class PlaybackService extends android.app.Service {

    private static final String CHANNEL_ID = "lp_playback";
    private static final int NOTIFICATION_ID = 1310;

    @Override
    public void onCreate() {
        super.onCreate();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "Playback",
                NotificationManager.IMPORTANCE_LOW // silent — it's a keep-alive, not an alert
            );
            channel.setDescription("Keeps audio playing in the background");
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) manager.createNotificationChannel(channel);
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Notification notification = buildNotification();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            // Android 14+ requires the type at startForeground time, matching the manifest.
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
        // START_NOT_STICKY: if the OS kills us, don't resurrect a service with no live playback.
        return START_NOT_STICKY;
    }

    private Notification buildNotification() {
        Intent launch = getPackageManager().getLaunchIntentForPackage(getPackageName());
        PendingIntent contentIntent = PendingIntent.getActivity(
            this,
            0,
            launch,
            PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
        );
        return new NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Learning Player")
            .setContentText("Playing in the background")
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentIntent(contentIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build();
    }

    /** Start the keep-alive foreground service. */
    public static void start(Context context) {
        Intent intent = new Intent(context, PlaybackService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(intent);
        } else {
            context.startService(intent);
        }
    }

    /** Stop the keep-alive foreground service. */
    public static void stop(Context context) {
        context.stopService(new Intent(context, PlaybackService.class));
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
