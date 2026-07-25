package app.closelistening.player;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        // Register the local background-audio plugin before the bridge starts (#1310).
        registerPlugin(BackgroundAudioPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
