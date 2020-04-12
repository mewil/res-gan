# gan

Creating the dataset

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y imagemagick

cd data/
mkdir train val test
# Download and unzip the data
aws s3 cp s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_train_AOI_1_Rio_3band.tar.gz .
aws s3 cp s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_test_AOI_1_Rio_3band.tar.gz .

tar -xvzf SN1_buildings_train_AOI_1_Rio_3band.tar.gz -C train/
tar -xvzf SN1_buildings_test_AOI_1_Rio_3band.tar.gz -C test/

rm SN1_buildings_train_AOI_1_Rio_3band.tar.gz
rm SN1_buildings_test_AOI_1_Rio_3band.tar.gz

# Convert the data to png form and crop from 438x406 to 406x406
cd train
for file in *.tif; do convert $file -chop 32x0 ${file%.tif}.png; done
for file in $(ls  *.png | tail -n 2795); do mv $file ../val/; done

cd ../test
for file in *.tif; do convert $file -chop 32x0 ${file%.tif}.png; done
```