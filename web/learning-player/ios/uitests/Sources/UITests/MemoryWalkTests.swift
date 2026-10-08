import XCTest

/// The iOS half of the device performance scan — `make perf-ios` (2026-10-08).
///
/// iOS has no DevTools protocol into a WKWebView, so the measuring happens OUTSIDE: this test walks
/// the app screen by screen (deep links, with the system's "Open in …?" confirm handled by
/// `AppSession.openLink`) and prints `=====MEM_STEP <name>=====`, then holds the screen still while
/// the host script reads the simulator's WebKit content process with `footprint`.
///
/// Environment (forwarded with the `TEST_RUNNER_` prefix on the xcodebuild PROCESS, not as build
/// settings — a build setting never reaches the runner):
///   LP_TOKEN    a session token: signs the app in on whatever backend it points at (prod included)
///   LP_WALK     comma-separated deep-link paths, e.g. `episode/<slug>,topic/topic:x`
///   LP_HOLD_S   seconds to hold each screen for the sample (default 12)
final class MemoryWalkTests: XCTestCase {
  func testWalkAndHold() {
    let env = ProcessInfo.processInfo.environment
    let hold = UInt32(env["LP_HOLD_S"] ?? "") ?? 12
    let app = XCUIApplication(bundleIdentifier: AppUnderTest.bundleId)
    app.launch()
    _ = app.wait(for: .runningForeground, timeout: 30)
    sleep(8)
    if let token = env["LP_TOKEN"], !token.isEmpty {
      AppSession.openLink(app, "auth#token=\(token)")
      sleep(8)
    }
    step("home", hold)
    let walk = (env["LP_WALK"] ?? "").split(separator: ",").map { String($0) }.filter { !$0.isEmpty }
    for path in walk {
      AppSession.openLink(app, path)
      sleep(4)
      step(path, hold)
    }
    print("=====MEM_DONE=====")
  }

  private func step(_ name: String, _ hold: UInt32) {
    print("=====MEM_STEP \(name)=====")
    sleep(hold)
  }
}
