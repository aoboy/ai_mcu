#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Run `west build -t run` for a Zephyr QEMU build under a time limit and make
# Ctrl-C stop the emulator promptly.
#
# `timeout` puts the west -> ninja -> QEMU chain in a dedicated process group.
# That keeps QEMU's stdio chardev off the controlling TTY (reading it from a
# background process group raises SIGTTIN and silently freezes the console),
# but it also means the terminal's SIGINT reaches neither QEMU nor `timeout`:
# only make, tee and this script are left in the foreground group. This wrapper
# therefore catches SIGINT/SIGTERM and forwards them to `timeout`, which
# signals the whole group so QEMU shuts down gracefully. QEMU's `-pidfile` is
# used as a fallback in case the process group is already gone.
#
# Usage: run_qemu.sh WEST_TOPDIR BUILD_DIR TIMEOUT_SECONDS

set -u

if [ "$#" -ne 3 ]; then
	echo "usage: $0 WEST_TOPDIR BUILD_DIR TIMEOUT_SECONDS" >&2
	exit 2
fi

west_topdir=$1
build_dir=$2
limit=$3
child=

# west resolves a relative -d against the workspace; make the build path
# absolute before this script leaves the caller's working directory.
case $west_topdir in
/*) ;;
*) west_topdir=$(cd "$west_topdir" && pwd) || exit 1 ;;
esac
case $build_dir in
/*) ;;
*) build_dir=$west_topdir/$build_dir ;;
esac
pid_file=$build_dir/qemu.pid

qemu_pid() {
	[ -r "$pid_file" ] || return 1
	pid=$(sed -n '1p' "$pid_file" 2>/dev/null) || return 1
	case $pid in
	'' | *[!0-9]*) return 1 ;;
	esac
	# The pidfile is ours (removed before the run), but do not signal a
	# recycled PID if QEMU died without cleaning it up.
	case $(cat "/proc/$pid/comm" 2>/dev/null) in
	qemu-system-*) ;;
	*) return 1 ;;
	esac
	printf '%s\n' "$pid"
}

stop_qemu() {
	qpid=$(qemu_pid) || return 0
	kill -0 "$qpid" 2>/dev/null || return 0

	# SIGINT is QEMU's graceful shutdown request; TERM and KILL are the
	# fallbacks if the emulator does not exit in time.
	kill -INT "$qpid" 2>/dev/null || return 0
	i=0
	while kill -0 "$qpid" 2>/dev/null && [ "$i" -lt 30 ]; do
		sleep 0.1
		i=$((i + 1))
	done
	kill -TERM "$qpid" 2>/dev/null || true
	i=0
	while kill -0 "$qpid" 2>/dev/null && [ "$i" -lt 20 ]; do
		sleep 0.1
		i=$((i + 1))
	done
	kill -KILL "$qpid" 2>/dev/null || true
}

stop_run() {
	if [ -n "$child" ]; then
		# timeout forwards SIGINT to its process group (west, ninja and
		# QEMU), so QEMU sees the same graceful shutdown as a terminal
		# Ctrl-C.
		kill -INT "$child" 2>/dev/null || true
	fi
	# Also stop QEMU from its pidfile, in case timeout has already exited
	# or its signal never made it through the group.
	stop_qemu
	if [ -n "$child" ]; then
		wait "$child" 2>/dev/null || true
	fi
}

# Drop a stale pidfile before the traps can act on it.
rm -f "$pid_file"

trap 'stop_run; exit 130' INT
trap 'stop_run; exit 131' QUIT
trap 'stop_run; exit 143' TERM HUP

cd "$west_topdir" || exit 1

# stdin from /dev/null: under `timeout` QEMU runs in a background process
# group, so reading the controlling TTY would raise SIGTTIN and stop it.
timeout --kill-after=2 "$limit" \
	west build -d "$build_dir" -t run < /dev/null &
child=$!

wait "$child"
status=$?
stop_qemu
exit "$status"
