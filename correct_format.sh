for d in */
do 
    cd $d

    python ../../correct_jpeg.py

    count=`ls -1 *.gz 2>/dev/null | wc -l`
    if [ $count != 0 ]; then 
        gunzip *gz;
        echo $d $count 'bad jpeg format'
    fi

    count=`ls -1 *.gz 2>/dev/null | wc -l`
    if [ $count != 0 ]; then 
        ls *gz | while read f; do mv $f "${f: 0:-3}"; done
    fi

    cd ..
done

