export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD
rm ~/.cache/gstreamer-1.0/ -fr    
sudo service nvargus-daemon restart
