#!/usr/bin/env bash
# One-time system setup for HOLD_FINAL. Run once after every reboot:
#
#     python3 core.py status      # (any core.py command writes generated/tun.conf)
#     sudo ./setup.sh
#
# 1. stops the stock AMF/SMF/UPF and the old HOLD units: an SMF without a
#    DNN/tracking-area restriction wins NRF discovery over ours;
# 2. creates one TUN device per UPF, owned by the calling user, so the 48 UPFs
#    can run as ordinary processes.
set -euo pipefail
cd "$(dirname "$0")"
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
owner="${SUDO_USER:?run with sudo from your own account}"
[ -f generated/tun.conf ] || { echo "generated/tun.conf missing: run 'python3 core.py status' first"; exit 1; }

systemctl stop open5gs-amfd open5gs-smfd open5gs-upfd
systemctl stop 'hold-*.service' 'hold2-*.service' 2>/dev/null || true

n=0
while read -r dev addrs; do
    ip link del "$dev" 2>/dev/null || true
    ip tuntap add name "$dev" mode tun user "$owner"
    for a in $addrs; do ip addr add "$a" dev "$dev"; done
    ip link set "$dev" up
    n=$((n + 1))
done < generated/tun.conf
echo "stock AMF/SMF/UPF stopped, $n UPF devices ready (owner $owner)"
