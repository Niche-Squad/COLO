# crontab -e to edit the cron jobs
# crontab -l shows the current cron jobs
# >> output.log 2>&1 
# * * * * * command to be executed
# - - - - -
# | | | | |
# | | | | ----- Day of week (0 - 7) (Sunday=0 or 7)
# | | | ------- Month (1 - 12)
# | | --------- Day of month (1 - 31)
# | ----------- Hour (0 - 23)
# ------------- Minute (0 - 59)

*/30 7-20 * * *  cd "/home/niche/cowsformer/" && sh stream_video.sh 