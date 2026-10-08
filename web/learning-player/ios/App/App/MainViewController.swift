import UIKit
import Capacitor
import MetricKit
import WebKit
import os

/**
 * Capacitor bridge view controller (#1310). Capacitor auto-registers only the plugins listed in
 * capacitor.config.json's packageClassList (npm plugins) — an APP-EMBEDDED plugin like AuthSession
 * is never in that list, so it must be registered explicitly here, once the bridge is loaded.
 * The storyboard's initial view controller points at this class (was CAPBridgeViewController).
 */
class MainViewController: CAPBridgeViewController {
    /// Held strongly: `WKWebView.navigationDelegate` is weak.
    private var terminationTap: TerminationTap?

    override func capacitorDidLoad() {
        bridge?.registerPluginInstance(AuthSession())
        bridge?.registerPluginInstance(AppProcess())

        // Why the app ends (#2279). Each source writes to ExitLog; the web layer forwards the log
        // to /api/app/app-exits on its next launch and clears it.
        MXMetricManager.shared.add(ExitMetrics.shared)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: nil
        ) { _ in
            ExitLog.append(source: "memory_warning", reason: "memory_warning")
        }
        if let webView = webView, let inner = webView.navigationDelegate {
            let tap = TerminationTap(inner: inner)
            terminationTap = tap
            webView.navigationDelegate = tap
        }
    }
}

/**
 * The app's own record of why it, or its WebView, ended (#2279) — kept in UserDefaults so it
 * survives the process, capped so it cannot grow without bound while the server is unreachable.
 */
enum ExitLog {
    private static let key = "lp.exitLog"
    private static let cap = 50
    private static let lock = NSLock()

    static func append(source: String, reason: String, count: Int = 1, at: Date = Date()) {
        lock.lock()
        defer { lock.unlock() }
        var list = UserDefaults.standard.array(forKey: key) as? [[String: Any]] ?? []
        list.append([
            "source": source, "reason": reason, "count": count,
            "at": ISO8601DateFormatter().string(from: at),
        ])
        if list.count > cap { list.removeFirst(list.count - cap) }
        UserDefaults.standard.set(list, forKey: key)
    }

    static func peek() -> [[String: Any]] {
        lock.lock()
        defer { lock.unlock() }
        return UserDefaults.standard.array(forKey: key) as? [[String: Any]] ?? []
    }

    /// Remove the first `count` entries — the ones the caller has just delivered.
    static func clear(count: Int) {
        lock.lock()
        defer { lock.unlock() }
        var list = UserDefaults.standard.array(forKey: key) as? [[String: Any]] ?? []
        list.removeFirst(min(max(count, 0), list.count))
        UserDefaults.standard.set(list, forKey: key)
    }
}

/**
 * MetricKit's daily exit counts (#2279) — iOS's own answer to "why did the app end", including the
 * background evictions no crash reporter sees. Delivered at most once a day, on a real device only.
 */
final class ExitMetrics: NSObject, MXMetricManagerSubscriber {
    static let shared = ExitMetrics()

    func didReceive(_ payloads: [MXMetricPayload]) {
        for payload in payloads {
            guard let exits = payload.applicationExitMetrics else { continue }
            let bg = exits.backgroundExitData
            let fg = exits.foregroundExitData
            let rows: [(String, Int)] = [
                ("bg_normal", bg.cumulativeNormalAppExitCount),
                ("bg_memory_limit", bg.cumulativeMemoryResourceLimitExitCount),
                ("bg_cpu_limit", bg.cumulativeCPUResourceLimitExitCount),
                ("bg_memory_pressure", bg.cumulativeMemoryPressureExitCount),
                ("bg_bad_access", bg.cumulativeBadAccessExitCount),
                ("bg_abnormal", bg.cumulativeAbnormalExitCount),
                ("bg_illegal_instruction", bg.cumulativeIllegalInstructionExitCount),
                ("bg_watchdog", bg.cumulativeAppWatchdogExitCount),
                ("bg_locked_file", bg.cumulativeSuspendedWithLockedFileExitCount),
                ("bg_task_timeout", bg.cumulativeBackgroundTaskAssertionTimeoutExitCount),
                ("fg_normal", fg.cumulativeNormalAppExitCount),
                ("fg_memory_limit", fg.cumulativeMemoryResourceLimitExitCount),
                ("fg_bad_access", fg.cumulativeBadAccessExitCount),
                ("fg_abnormal", fg.cumulativeAbnormalExitCount),
                ("fg_illegal_instruction", fg.cumulativeIllegalInstructionExitCount),
                ("fg_watchdog", fg.cumulativeAppWatchdogExitCount),
            ]
            for (reason, count) in rows where count > 0 {
                ExitLog.append(source: "metrickit", reason: reason, count: count, at: payload.timeStampEnd)
            }
        }
    }
}

/**
 * Records the WebView's content process dying, then lets Capacitor handle it exactly as before.
 *
 * Capacitor creates its navigation delegate privately, so there is no subclass hook. This sits in
 * front of it: it implements ONE method and forwards every other selector to Capacitor's delegate
 * (`responds(to:)` + `forwardingTarget(for:)`), so WebKit sees the same delegate it always did.
 */
final class TerminationTap: NSObject, WKNavigationDelegate {
    private let inner: WKNavigationDelegate

    init(inner: WKNavigationDelegate) {
        self.inner = inner
    }

    override func responds(to aSelector: Selector!) -> Bool {
        super.responds(to: aSelector) || inner.responds(to: aSelector)
    }

    override func forwardingTarget(for aSelector: Selector!) -> Any? {
        inner.responds(to: aSelector) ? inner : nil
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        ExitLog.append(source: "webview_terminated", reason: "webcontent_terminated")
        inner.webViewWebContentProcessDidTerminate?(webView)
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
        CAPPluginMethod(name: "uptime", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "exitLog", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "clearExitLog", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "memoryInfo", returnType: CAPPluginReturnPromise),
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

    /// The pending exit records (#2279), oldest first. Not cleared until `clearExitLog`.
    @objc func exitLog(_ call: CAPPluginCall) {
        call.resolve(["entries": ExitLog.peek()])
    }

    /// For "Copy debug info" (operator 2026-10-08): how much memory iOS will still let this app use
    /// before it is killed, the app's own footprint, total RAM, thermal state and Low Power Mode. iOS
    /// gives no figure for free GPU memory; the WebContent process's memory is not in the footprint.
    @objc func memoryInfo(_ call: CAPPluginCall) {
        var result: [String: Any] = [
            "totalMb": Int(ProcessInfo.processInfo.physicalMemory / 1_048_576),
            "thermalState": ProcessInfo.processInfo.thermalState.rawValue,
            "lowPowerMode": ProcessInfo.processInfo.isLowPowerModeEnabled,
        ]
        if #available(iOS 13.0, *) {
            result["availMb"] = Int(os_proc_available_memory() / 1_048_576)
        }
        var vm = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &vm) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS { result["appFootprintMb"] = Int(vm.phys_footprint / 1_048_576) }
        var sys = utsname()
        uname(&sys)
        let machine = withUnsafePointer(to: &sys.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        result["model"] = machine
        result["osVersion"] = UIDevice.current.systemVersion
        call.resolve(result)
    }

    @objc func clearExitLog(_ call: CAPPluginCall) {
        ExitLog.clear(count: call.getInt("count") ?? 0)
        call.resolve()
    }
}
