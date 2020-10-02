ffmpeg -i /home/vpreston/Documents/IPP/informative-path-planning/TRIALS/iros_figures/experiments/sim_seed9500-pathsetdubins-nonmyopicTrue-treedpw-FREE/figures/mes_resized_images/trajectory-N.%d.png \
  -c:v libx264 -crf 23 -profile:v baseline -level 3.0 -pix_fmt yuv420p \
  -c:a aac -ac 2 -strict experimental -b:a 128k \
  -vf scale=1280:1280 -movflags faststart \
  example_video_square.mp4
