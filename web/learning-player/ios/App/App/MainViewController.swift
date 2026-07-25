import UIKit
import Capacitor

/**
 * Capacitor bridge view controller (#1310). Capacitor auto-registers only the plugins listed in
 * capacitor.config.json's packageClassList (npm plugins) — an APP-EMBEDDED plugin like AuthSession
 * is never in that list, so it must be registered explicitly here, once the bridge is loaded.
 * The storyboard's initial view controller points at this class (was CAPBridgeViewController).
 */
class MainViewController: CAPBridgeViewController {
    override func capacitorDidLoad() {
        bridge?.registerPluginInstance(AuthSession())
    }
}
