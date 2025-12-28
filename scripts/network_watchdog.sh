#!/bin/bash
# Network Watchdog for Raspberry Pi Wildlife Camera
# Monitors connectivity and takes corrective action on failures

# Configuration
PING_TARGET="8.8.8.8"                    # Primary target (Google DNS)
PING_TARGET_BACKUP="1.1.1.1"             # Backup target (Cloudflare DNS)
GATEWAY=""                                # Will be auto-detected
CHECK_INTERVAL=30                         # Seconds between checks
PING_TIMEOUT=5                            # Seconds to wait for ping
PING_COUNT=2                              # Number of pings per check
MAX_FAILURES_BEFORE_RESTART=3             # NetworkManager restart threshold
MAX_FAILURES_BEFORE_REBOOT=8              # System reboot threshold (after NM restart)
LOG_FILE="/var/log/network-watchdog.log"
STATE_FILE="/tmp/network-watchdog-failures"

# Telegram notification (optional - set these to enable)
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""

# Load .env file if it exists (for Telegram credentials)
ENV_FILE="/home/daniel/animal_tracker/.env"
if [[ -f "$ENV_FILE" ]]; then
    export $(grep -E "^TELEGRAM_(BOT_TOKEN|CHAT_ID)=" "$ENV_FILE" | xargs)
    if [[ -n "$TELEGRAM_BOT_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
        TELEGRAM_ENABLED=true
    fi
fi

# Logging function
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"

    # Also log to systemd journal
    logger -t network-watchdog "[$level] $message"
}

# Send Telegram notification
send_telegram() {
    local message="$1"
    if [[ "$TELEGRAM_ENABLED" == "true" ]]; then
        local hostname=$(hostname)
        local full_message="🌐 Network Watchdog [$hostname]%0A$message"
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            -d "text=${full_message}" \
            -d "parse_mode=HTML" \
            --max-time 10 > /dev/null 2>&1
    fi
}

# Get default gateway
get_gateway() {
    ip route | grep default | awk '{print $3}' | head -1
}

# Check connectivity
check_connectivity() {
    local target="$1"
    ping -c "$PING_COUNT" -W "$PING_TIMEOUT" "$target" > /dev/null 2>&1
    return $?
}

# Get WiFi signal strength
get_wifi_signal() {
    iwconfig wlan0 2>/dev/null | grep -oP 'Signal level=\K-[0-9]+' || echo "N/A"
}

# Get failure count from state file
get_failure_count() {
    if [[ -f "$STATE_FILE" ]]; then
        cat "$STATE_FILE"
    else
        echo "0"
    fi
}

# Set failure count
set_failure_count() {
    echo "$1" > "$STATE_FILE"
}

# Restart NetworkManager
restart_networkmanager() {
    log "WARN" "Restarting NetworkManager..."
    send_telegram "⚠️ Restarting NetworkManager due to connectivity issues"

    systemctl restart NetworkManager
    sleep 10

    # Wait for connection to come back up
    for i in {1..6}; do
        if check_connectivity "$PING_TARGET"; then
            log "INFO" "NetworkManager restart successful - connectivity restored"
            send_telegram "✅ NetworkManager restart successful - connectivity restored"
            return 0
        fi
        sleep 5
    done

    log "ERROR" "NetworkManager restart did not restore connectivity"
    return 1
}

# Restart WiFi interface
restart_wifi_interface() {
    log "WARN" "Restarting WiFi interface..."

    nmcli radio wifi off
    sleep 2
    nmcli radio wifi on
    sleep 10

    # Force reconnection
    nmcli connection up preconfigured 2>/dev/null || true
    sleep 5
}

# Reboot system
reboot_system() {
    log "CRIT" "Initiating system reboot due to persistent network failure"
    send_telegram "🔄 Rebooting system due to persistent network failure"

    # Give Telegram message time to send
    sleep 2

    sync
    /sbin/reboot
}

# Main watchdog loop
main() {
    log "INFO" "Network watchdog started"
    log "INFO" "Check interval: ${CHECK_INTERVAL}s, NM restart after: ${MAX_FAILURES_BEFORE_RESTART} failures, Reboot after: ${MAX_FAILURES_BEFORE_REBOOT} failures"

    # Reset failure count on start
    set_failure_count 0

    # Auto-detect gateway
    GATEWAY=$(get_gateway)
    if [[ -n "$GATEWAY" ]]; then
        log "INFO" "Detected gateway: $GATEWAY"
    fi

    # Wait for network to be ready, then send startup notification
    sleep 5
    local wifi_signal=$(get_wifi_signal)
    local ip_addr=$(ip -4 addr show wlan0 2>/dev/null | grep -oP 'inet \K[\d.]+' || echo "unknown")
    send_telegram "🟢 System started%0AIP: ${ip_addr}%0ASignal: ${wifi_signal} dBm%0AGateway: ${GATEWAY}"
    log "INFO" "Startup notification sent (IP: ${ip_addr}, Signal: ${wifi_signal} dBm)"

    local last_status="up"
    local consecutive_successes=0

    while true; do
        local failure_count=$(get_failure_count)
        local wifi_signal=$(get_wifi_signal)
        local connectivity_ok=false

        # Try gateway first, then external targets
        if [[ -n "$GATEWAY" ]] && check_connectivity "$GATEWAY"; then
            connectivity_ok=true
        elif check_connectivity "$PING_TARGET"; then
            connectivity_ok=true
        elif check_connectivity "$PING_TARGET_BACKUP"; then
            connectivity_ok=true
        fi

        if $connectivity_ok; then
            # Connection is up
            if [[ "$last_status" == "down" ]]; then
                log "INFO" "Connectivity restored (signal: ${wifi_signal} dBm)"
                send_telegram "✅ Connectivity restored (signal: ${wifi_signal} dBm)"
            fi

            # Reset failure count after sustained success
            consecutive_successes=$((consecutive_successes + 1))
            if [[ $consecutive_successes -ge 2 ]]; then
                set_failure_count 0
            fi

            last_status="up"

            # Log signal strength periodically (every 10 minutes = 20 checks at 30s interval)
            if [[ $((RANDOM % 20)) -eq 0 ]]; then
                log "DEBUG" "Connection healthy, signal: ${wifi_signal} dBm"
            fi
        else
            # Connection is down
            consecutive_successes=0
            failure_count=$((failure_count + 1))
            set_failure_count "$failure_count"

            if [[ "$last_status" == "up" ]]; then
                log "WARN" "Connectivity lost! (failure #${failure_count}, signal was: ${wifi_signal} dBm)"
                send_telegram "❌ Connectivity lost! (failure #${failure_count})"
            else
                log "WARN" "Connectivity still down (failure #${failure_count})"
            fi

            last_status="down"

            # Take corrective action based on failure count
            if [[ $failure_count -ge $MAX_FAILURES_BEFORE_REBOOT ]]; then
                reboot_system
                # Won't reach here
            elif [[ $failure_count -eq $MAX_FAILURES_BEFORE_RESTART ]]; then
                restart_wifi_interface
            elif [[ $failure_count -eq $((MAX_FAILURES_BEFORE_RESTART + 2)) ]]; then
                restart_networkmanager
            fi
        fi

        sleep "$CHECK_INTERVAL"
    done
}

# Handle signals
trap 'log "INFO" "Network watchdog stopped"; exit 0' SIGTERM SIGINT

# Run main loop
main
