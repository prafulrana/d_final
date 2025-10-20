#!/bin/bash
# Manage local system (frpc + containers)

case "$1" in
    start)
        ./frpc-start.sh && ./start.sh
        ;;
    stop)
        ./stop.sh && ./frpc-stop.sh
        ;;
    restart)
        ./stop.sh && ./frpc-restart.sh && ./start.sh
        ;;
    status)
        ./check.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
