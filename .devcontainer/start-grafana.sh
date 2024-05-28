#!/bin/bash
# start-grafana.sh

# Start Grafana server
sudo -u grafana /usr/sbin/grafana-server \
    --homepath=/usr/share/grafana \
    --config=/etc/grafana/grafana.ini \
    cfg:default.paths.logs=/var/log/grafana \
    cfg:default.paths.data=/var/lib/grafana \
    cfg:default.paths.plugins=/var/lib/grafana/plugins \
    >/dev/null 2>&1 &
# Sleep for a few seconds to ensure Grafana has started
while true; do
    if curl -sSf http://localhost:3000/login >/dev/null 2>&1; then
        # Grafana is accessible, print a message and exit the script
        echo "Grafana has started successfully."
        exit 0
    fi
    # Wait for a few seconds before checking again
    sleep 5
done
# Print a message indicating that Grafana has started
echo "Grafana has started successfully."

# Keep the script running to prevent it from exiting
tail -f /dev/null