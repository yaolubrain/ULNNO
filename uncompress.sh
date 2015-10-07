source /home/fs/ylu/.bashrc
ls *.tar | cut -c1-9 | xargs mkdir -vp

for f in *.tar
do
	tar -xvf $f -C ${f:0:9}/
	echo 'uncompressed file '$f' to folder '${f:0:9}/
done
