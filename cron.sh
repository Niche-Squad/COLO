# crontab -e to edit the cron jobs
# crontab -l shows the current cron jobs
*/10 7-20 * * *  cd "/Users/niche/OneDrive - Virginia Tech/_03_Papers/2023/cowsformer/" && sh stream_video.sh >> output.log 2>&1