#!/bin/sh
# backup_server_launcher.sh
# launches correct python scripts with directory management

cd ~/pinpoint
sleep 10
python3 -u photo_data_analysis.py > logs 2>&1 &
exit 0
