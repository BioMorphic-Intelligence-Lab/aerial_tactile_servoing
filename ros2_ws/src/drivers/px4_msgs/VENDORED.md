# Vendored dependency — do not blindly update

This directory is a **vendored snapshot** of [PX4/px4_msgs](https://github.com/PX4/px4_msgs).
It was formerly a git submodule pinned to commit
`c3571707eed2d6b3661d36e0a338152fdb6795fd`, and was absorbed into this repository (originally
by commit `b05ca5e`, 2025-11-04).

**Firmware coupling:** these uORB message definitions must match the PX4 firmware running on the
Pixhawk — **v1.16.0** (see `px4_params/fc_params_vbats.params`). If they drift from the
flight-controller firmware, the uXRCE-DDS bridge mismatches silently.

**Do not update these files unless you reflash the Pixhawk to a matching PX4 version.**
