for d in */
do 
    ls $d*.JPEG > ${d:0:9}.txt
    echo 'created image list '${d:0:9}.txt
done


IM2H5=/home/fs/ylu/Code/convnet/bin/image2hdf5
for l in *.txt
do
    $IM2H5 --input=$l --output=${l:0:9}.h5 --resize=256 --crop=224
    echo 'created hdf5 images '${l:0:9}.h5
done
