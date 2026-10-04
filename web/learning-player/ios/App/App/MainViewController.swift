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
        bridge?.registerPluginInstance(AppProcess())
    }
}

/**
 * AppProcess (#2277) — how long THIS app process has been alive.
 *
 * The web layer cannot tell a cold launch (iOS ended the whole app; the native splash shows) from a
 * WebView reload (iOS ended only the WebContent process and Capacitor reloaded the page; the web
 * splash shows). Both look like "starting from scratch". A page that boots inside a process much
 * older than itself is a reload. Lives in this file because a new Swift file would need a
 * hand-edited project.pbxproj entry.
 */
@objc(AppProcess)
public class AppProcess: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "AppProcess"
    public let jsName = "AppProcess"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "uptime", returnType: CAPPluginReturnPromise)
    ]

    /** The kernel's start time for this pid; read once, it never changes. */
    private static let started: Date? = {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        guard sysctl(&mib, u_int(mib.count), &info, &size, nil, 0) == 0 else { return nil }
        let tv = info.kp_proc.p_un.__p_starttime
        return Date(timeIntervalSince1970: Double(tv.tv_sec) + Double(tv.tv_usec) / 1_000_000)
    }()

    @objc func uptime(_ call: CAPPluginCall) {
        guard let started = AppProcess.started else {
            call.reject("process start time unavailable")
            return
        }
        call.resolve(["ms": Int(Date().timeIntervalSince(started) * 1000)])
    }
}
