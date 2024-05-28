FROM mcr.microsoft.com/devcontainers/anaconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp
# Ensure sudo is installed
RUN apt-get update && \
    apt-get install -y sudo && \
    rm -rf /var/lib/apt/lists/*

# Add vscode user to sudoers without requiring password
RUN echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode
# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
# Install Grafana
RUN apt-get update && \
    apt-get install -y apt-transport-https software-properties-common wget && \
    wget -q -O - https://packages.grafana.com/gpg.key | apt-key add - && \
    add-apt-repository "deb https://packages.grafana.com/oss/deb stable main" && \
    apt-get update && \
    apt-get install -y grafana
#aaaaaaaaaaaaaaaa
RUN chown -R grafana:grafana /etc/grafana /var/lib/grafana /var/log/grafana && \
    chmod -R 755 /etc/grafana /var/lib/grafana /var/log/grafana
# Enable Grafana to start on boot and expose the default Grafana port
# Copy the startup script
COPY .devcontainer/start-grafana.sh /usr/local/bin/start-grafana.sh

# Grant execute permissions to the startup script
RUN chmod +x /usr/local/bin/start-grafana.sh

# Expose Grafana port
EXPOSE 3000

# Define the command to run when the container starts
CMD ["/usr/local/bin/start-grafana.sh"]