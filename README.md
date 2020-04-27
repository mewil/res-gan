# Residual cGANs for Cloud Cover Removal

Final Project for University of Michigan EECS 442. Code adapted from [Enomoto et al](https://github.com/enomotokenji/mcgan-cvprw2017-pytorch) and [Python-Clouds](https://github.com/SquidDev/Python-Clouds).

Downloading and generating the dataset:

```sh
# AWS CLI must be installed
cd data/
mkdir images
# Download and unzip the data
aws s3 cp s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_train_AOI_1_Rio_3band.tar.gz .
aws s3 cp s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_test_AOI_1_Rio_3band.tar.gz .

tar -xvzf SN1_buildings_train_AOI_1_Rio_3band.tar.gz -C images/
tar -xvzf SN1_buildings_test_AOI_1_Rio_3band.tar.gz -C images/

rm SN1_buildings_train_AOI_1_Rio_3band.tar.gz
rm SN1_buildings_test_AOI_1_Rio_3band.tar.gz

# Then split the data into train, validation, and test directories before running the cloud generation script
python3 generate_clouds.py --input_dir train
python3 generate_clouds.py --input_dir validation
python3 generate_clouds.py --input_dir test
```
