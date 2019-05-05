ffmpeg -i ../experiments/nonmyopic_01_seed10000/figures/mean/trajectory-N.%d.png \
  -c:v libx264 -crf 23 -profile:v baseline -level 3.0 -pix_fmt yuv420p \
  -c:a aac -ac 2 -strict experimental -b:a 128k \
  -movflags faststart \
  seed10000_ucb.mp4
