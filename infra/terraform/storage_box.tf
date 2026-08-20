# #1199 — remote audio archive on a cloud-native Hetzner Storage Box.
#
# The pipeline persists raw episode audio here (audio_storage_backend='remote')
# so reprocessing reads from the archive instead of re-fetching feeds. A Storage
# Box is object-style storage at ~EUR3.20/mo for 1 TB (bx11) — ~18x cheaper than
# a Cloud Volume for a cold, reprocess-only archive (see the wip sizing doc).
#
# Gated on audio_storage_box_type: empty (default) provisions nothing, so PROD is
# unaffected until the operator opts in via tfvars. rclone reaches it over SFTP
# (the SSH subsystem) using the exported server/username + the password below.
resource "hcloud_storage_box" "audio_archive" {
  count            = var.audio_storage_box_type != "" ? 1 : 0
  name             = "podcast-audio-archive"
  storage_box_type = var.audio_storage_box_type
  location         = var.audio_storage_box_location
  password         = var.audio_storage_box_password

  # SFTP (used by rclone) runs over the SSH subsystem. Keep the box reachable
  # from the VPS and the operator's laptop; leave Samba/WebDAV/ZFS off.
  #
  # SECURITY (review 2026-07-17 low/storage-box): reachable_externally=true makes
  # this SFTP endpoint INTERNET-FACING with password-only auth (the Storage Box is
  # a separate Hetzner product — the VPS firewall does NOT protect it, and it has
  # no server-side IP allowlist). Mitigation: make audio_storage_box_password
  # high-entropy (>=32 random chars). Consider key-only auth (add ssh_keys) once
  # the operator pre-registers a key with Hetzner.
  access_settings = {
    ssh_enabled          = true
    reachable_externally = true
    samba_enabled        = false
    webdav_enabled       = false
    zfs_enabled          = false
  }

  # Same delete ratchet as the corpus volume: the archive is expensive to rebuild
  # (would require re-downloading every episode). PROD keeps this true; the drill
  # workspace sets enable_delete_protection=false so its teardown can destroy.
  delete_protection = var.enable_delete_protection

  # Weekly snapshot of the archive as cheap insurance (Sunday 03:00 UTC, keep 4).
  snapshot_plan = {
    hour          = 3
    minute        = 0
    day_of_week   = 0
    max_snapshots = 4
  }
}

# #1787 (advisor M5) — a sub-account scoped to the archive path, so the credentials the prod
# pipeline holds in its plaintext .env (RCLONE_CONFIG_AUDIOARCHIVE_*, delete-capable now that
# eviction is on) are JAILED to podcast-audio-archive/ instead of the whole box. Blast-radius
# containment on a .env leak: the jail blocks traversal to anything else on the box, and the
# box's snapshots (managed at the account level, outside this sub-account's SFTP view) cannot be
# deleted with these creds — so the archive stays recoverable even if every file in the jail is
# wiped. THREAT_MODEL T-08 addendum.
#
# The pipeline uses THESE creds (username exported below), not the main account's; the sub-account
# is home-dir'd to podcast-audio-archive/, so the rclone base_path is empty (the jail root IS that
# dir). Gated on the sub-account password so the box can exist without it (staged migration).
resource "hcloud_storage_box_subaccount" "audio_archive" {
  count          = (var.audio_storage_box_type != "" && var.audio_storage_box_subaccount_password != "") ? 1 : 0
  storage_box_id = hcloud_storage_box.audio_archive[0].id
  home_directory = "podcast-audio-archive/"
  password       = var.audio_storage_box_subaccount_password

  # Read-WRITE (uploads + eviction-side deletes within the jail need it); SFTP only, same
  # internet-facing caveat as the parent box (mitigated by a high-entropy password).
  access_settings = {
    ssh_enabled          = true
    reachable_externally = true
    samba_enabled        = false
    webdav_enabled       = false
    readonly             = false
  }

  labels = {
    purpose = "podcast-audio-archive-pipeline"
  }
}
