#!/usr/bin/env bash
# Give the build account a signing keychain, so `fastlane beta` can codesign (#2189).
#
# WHY THIS EXISTS
#
# The `claude` account on the build Mac has never had a GUI login, so macOS never created its login
# keychain. `security default-keychain` reports "A default keychain could not be found", and
# `security find-identity -v -p codesigning` reports "0 valid identities found" — the Apple
# certificates live in the operator's own login keychain, which this account cannot see.
#
# Xcode's automatic signing tries to INSTALL a certificate into a keychain, so with none present the
# archive dies with:
#
#     error: Certificate installation failed: ... "A default keychain could not be found."
#     error: No profiles for 'app.closelistening.player' were found
#
# The second error is a consequence of the first, not a separate problem: profile lookup fails
# because the certificate never landed.
#
# This is the standard CI setup for exactly this situation — a dedicated keychain holding only the
# signing identity, unlocked for the session.
#
# USAGE
#
#   scripts/tools/setup_signing_keychain.sh /path/to/signing.p12 /path/to/p12-password.txt
#
# The .p12 must contain the certificate AND its private key (export from Keychain Access via
# login -> My Certificates, expanding the certificate to confirm a key is under it).
#
# Safe to re-run: the keychain is recreated from scratch each time.
set -euo pipefail

P12="${1:?usage: $0 <signing.p12> <p12-password-file>}"
PW_FILE="${2:?usage: $0 <signing.p12> <p12-password-file>}"

KEYCHAIN="${SIGNING_KEYCHAIN:-$HOME/Library/Keychains/ios-signing.keychain-db}"
KEYCHAIN_NAME="$(basename "$KEYCHAIN")"
# The keychain's own password. Not a secret in any meaningful sense — it protects a keychain that
# exists only on this machine, holding a certificate that is useless without Apple's account. It is
# generated rather than hardcoded so it is at least not a published constant.
KC_PW="${SIGNING_KEYCHAIN_PASSWORD:-$(LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 32)}"

[ -r "$P12" ] || { echo "FAIL: cannot read $P12"; exit 1; }
[ -r "$PW_FILE" ] || { echo "FAIL: cannot read $PW_FILE"; exit 1; }
P12_PW="$(tr -d '\r\n' < "$PW_FILE")"

echo "--> recreating $KEYCHAIN_NAME"
security delete-keychain "$KEYCHAIN" 2>/dev/null || true
security create-keychain -p "$KC_PW" "$KEYCHAIN"
# No auto-lock. A keychain that re-locks mid-archive fails the build halfway through, which is a
# far more confusing failure than not being set up at all.
security set-keychain-settings -lut 21600 "$KEYCHAIN"
security unlock-keychain -p "$KC_PW" "$KEYCHAIN"

echo "--> importing the signing identity"
# -A would let ANY application use the key without prompting. Scoped to the tools that actually
# sign instead, which is the same posture a developer's own keychain takes.
security import "$P12" -k "$KEYCHAIN" -P "$P12_PW" \
    -T /usr/bin/codesign -T /usr/bin/security -T /usr/bin/productsign

echo "--> setting the partition list"
# WITHOUT THIS, codesign blocks on a GUI dialog asking permission to use the key — on an account
# with no GUI session, which means the build hangs until it times out. This is the single most
# common way a CI signing setup looks correct and does not work.
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$KC_PW" "$KEYCHAIN" >/dev/null

echo "--> putting it in the search list, as the default"
# Keep the System keychain in the list: dropping it breaks TLS trust evaluation for anything else
# running as this user.
security list-keychains -d user -s "$KEYCHAIN" /Library/Keychains/System.keychain
security default-keychain -s "$KEYCHAIN"

echo
echo "--> identities now visible:"
security find-identity -v -p codesigning "$KEYCHAIN"

cat <<EOF

Keychain password (needed to re-unlock after a reboot — store it or re-run this script):

    $KC_PW

If a build later fails with "User interaction is not allowed", the keychain has locked. Unlock it:

    security unlock-keychain -p '$KC_PW' "$KEYCHAIN"
EOF
