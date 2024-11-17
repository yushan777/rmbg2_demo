# clone this repo
git clone https://github.com/yushan777/rmbg2_demo.git
cd rmbg2_demo

# run installer
chmod +x install.sh
./install.sh

# activate environment
source venv/bin/activate

# remove background from a single image
python3 rmbg.py -i 'path/to/image.png'

# remove background from images in a directory 
python3 rmbg_batch.py -i 'path/to/dir_of_images'

