cd ./local_data/

mkdir -p ./pix3d ./StanfordCars ./CompCars ./experiments

mv pix3d.zip ./pix3d/ && cd ./pix3d/ && unzip pix3d.zip && cd ..
mv pix3d_test_list.txt ./pix3d/
mv cars_train.tgz cars_test.tgz ./StanfordCars/ && cd ./StanfordCars/ && tar xvzf cars_train.tgz && tar xvzf cars_test.tgz  && cd ..
mv CompCars.zip ./CompCars/ && cd ./CompCars/ && unzip CompCars.zip
zip -F data.zip --out combined.zip && unzip -P d89551fd190e38 combined.zip && cd ..

mv ./finegrained-pose/Anno3D/StanfordCars/* ./StanfordCars/
mv ./finegrained-pose/CAD/StanfordCars3D.txt ./StanfordCars/

mv ./finegrained-pose/Anno3D/CompCars/* ./CompCars/
mv ./finegrained-pose/CAD/CompCars3D.txt ./CompCars/

mkdir -p ./CompCars/models ./StanfordCars/models

unzip 02958343.zip
cd 02958343
for file in $(<../CompCars/CompCars3D.txt); do cp -r "$file" ../CompCars/models/; done
for file in $(<../StanfordCars/StanfordCars3D.txt); do cp -r "$file" ../StanfordCars/models/; done
cd ..

tar -xvf VOCtrainval_11-May-2012.tar
rm -rf ./VOCdevkit/VOC2012/Annotations ./VOCdevkit/VOC2012/ImageSets ./VOCdevkit/VOC2012/Segmentation*

mkdir ./texture_datasets
unzip textures.zip -d ./texture_datasets

unzip checkpoints.zip -d ./experiments/

cd ..