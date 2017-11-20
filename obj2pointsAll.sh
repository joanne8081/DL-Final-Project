for file in ./data/*/*.obj
do
    OBJFILE="${file}"
    PCFILE="${file/.obj/.txt}"
    cp $OBJFILE $PCFILE
    sed -i.bak '/^v /! d' $PCFILE
    sed -i.bak 's/^v //g' $PCFILE
    rm $PCFILE.bak
    #sh obj2points.sh file "${file/.obj/.txt}"
done
