import Foundation
import Capacitor
import AuthenticationServices

/**
 * AuthSession (#1310) — prompt-free native OAuth return on iOS.
 *
 * `@capacitor/browser` + a custom-scheme deep link makes iOS show an "Open in <app>?" confirmation
 * on the callback (custom-scheme app handoff). ASWebAuthenticationSession is Apple's purpose-built
 * OAuth primitive: it presents the auth page and, when the page redirects to `callbackScheme://…`,
 * the OS captures it and returns the URL directly to the completion handler — no dialog, no global
 * deep-link listener. Android returns prompt-free via its intent-filter, so this is iOS-only.
 *
 * App-embedded plugins are NOT in capacitor.config.json's packageClassList, so Capacitor's
 * auto-registration never loads them — this instance is registered explicitly in
 * MainViewController.capacitorDidLoad(). CAPBridgedPlugin is required by registerPluginInstance().
 */
@objc(AuthSession)
public class AuthSession: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "AuthSession"
    public let jsName = "AuthSession"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "start", returnType: CAPPluginReturnPromise)
    ]

    // Held for the session's lifetime so it isn't deallocated mid-flight.
    private var session: ASWebAuthenticationSession?

    @objc func start(_ call: CAPPluginCall) {
        guard let urlString = call.getString("url"), let url = URL(string: urlString) else {
            call.reject("A valid 'url' is required")
            return
        }
        guard let scheme = call.getString("callbackScheme") else {
            call.reject("A 'callbackScheme' is required")
            return
        }
        DispatchQueue.main.async {
            let session = ASWebAuthenticationSession(url: url, callbackURLScheme: scheme) { callbackURL, error in
                if let error = error {
                    let nsError = error as NSError
                    if nsError.domain == ASWebAuthenticationSessionError.errorDomain
                        && nsError.code == ASWebAuthenticationSessionError.canceledLogin.rawValue {
                        call.reject("cancelled", "CANCELLED")
                    } else {
                        call.reject("auth session failed: \(error.localizedDescription)")
                    }
                    return
                }
                call.resolve(["url": callbackURL?.absoluteString ?? ""])
            }
            session.presentationContextProvider = self
            // Ephemeral: no Safari cookie sharing → no one-time "wants to use X to sign in" consent
            // (#1310, operator's call). Trade-off: no Google SSO — but we mint our own 30-day token so
            // real OAuth logins are rare, and a dialog-free flow is preferred. Flip to false for SSO.
            session.prefersEphemeralWebBrowserSession = true
            self.session = session
            session.start()
        }
    }
}

extension AuthSession: ASWebAuthenticationPresentationContextProviding {
    public func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        if let window = self.bridge?.viewController?.view.window {
            return window
        }
        // Fallback: the app's current key window (bridge VC window can be nil mid-launch).
        let keyWindow = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .flatMap { $0.windows }
            .first { $0.isKeyWindow }
        return keyWindow ?? ASPresentationAnchor()
    }
}
